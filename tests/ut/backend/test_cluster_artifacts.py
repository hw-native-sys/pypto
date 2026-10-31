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
import copy
import ctypes
import gzip
import hashlib
import importlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pypto.language as pl
import pytest
import torch
from pypto import backend, codegen, ir
from pypto.backend import BackendType, pto_backend
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit._artifact_manifest import BuildKind
from pypto.runtime import _artifact_sources, _prebuilt


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
        assert entry["layout"]["version"] == 1
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


_RUNTIME_IMPORT_PROOF = r"""
import importlib
import importlib.abc
import sys
class BlockPyPTO(importlib.abc.MetaPathFinder):
    attempts = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pypto' or fullname.startswith('pypto.'):
            self.attempts.append(fullname)
            raise AssertionError('Runtime sources imported ' + fullname)
guard = BlockPyPTO()
sys.meta_path.insert(0, guard)
namespace, digest, archive_path, cache_root = sys.argv[1:]
from pathlib import Path
from simpler.startup_artifacts import (
    StartupArtifactSpec, read_startup_artifacts, install_startup_artifacts, activate_startup_artifacts,
)
assert not Path(cache_root).exists()
spec = StartupArtifactSpec(archive_path, digest, namespace)
descriptions = read_startup_artifacts([spec.to_manifest()])
installed = install_startup_artifacts(descriptions, cache_root)
assert len(installed) == 1 and installed[0].digest == digest
activate_startup_artifacts(installed)
helper = importlib.import_module(namespace + '.orchestration._distributed_support')
for name in ('first', 'second'):
    module = importlib.import_module(namespace + '.next_levels.' + name + '.orchestration.host_orch')
    entry = getattr(module, name)
    assert entry._pypto_distributed_entry and entry._pypto_embedded_layout['version'] == 1
    shim = importlib.import_module(namespace + '.next_levels.' + name + '.orchestration')
    assert shim._distributed_support is helper
observer = importlib.import_module(namespace + '.next_levels.first.sub_workers.observe')
from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker
owner = mint_owner_instance_id()
buffers = [create_host_shared_buffer(32, owner, index + 1) for index in range(2)]
worker = None
try:
    buffers[0].shm.buf[:] = bytes(32)
    buffers[1].shm.buf[:] = bytes([0xcd]) * 32
    source = buffers[0].tensor(shapes=(8,), dtype=DataType.FLOAT32)
    destination = buffers[1].tensor(shapes=(8,), dtype=DataType.FLOAT32)
    worker = Worker(level=3, num_sub_workers=2)
    observe_handle = worker.register(observer.observe)
    copy_handle = worker.register(helper._copy_return_tensor)
    worker.init()
    def orchestration(orch, args, config):
        observation = TaskArgs()
        observation.add_tensor(source, TensorArgType.INPUT)
        orch.submit_sub(observe_handle, observation)
        copying = TaskArgs()
        copying.add_tensor(source, TensorArgType.INPUT)
        copying.add_tensor(destination, TensorArgType.OUTPUT_EXISTING)
        orch.submit_sub(copy_handle, copying)
    worker.run(orchestration, args=None, config=CallConfig())
    assert bytes(buffers[1].shm.buf) == bytes(32)
finally:
    if worker is not None:
        worker.close()
    for backing in buffers:
        backing.close()
assert not guard.attempts
assert not any(name == 'pypto' or name.startswith('pypto.') for name in sys.modules)
print('C3C_RUNTIME_SOURCES_PASS')
"""


@pytest.fixture
def cluster_runner_harness(cluster_artifact, monkeypatch):
    """Keep native generated root/metadata and replace only remote process/device I/O."""
    runner = importlib.import_module("pypto.runtime.cluster_runner")
    from simpler.callable_identity import CallableHandle  # noqa: PLC0415
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
            state.worker = self

        def register(self, target, *, workers=None):
            registrations.append((target, workers))
            remote = isinstance(target, RemoteCallable)
            return CallableHandle(
                "sha256:" + hashlib.sha256(str(len(registrations)).encode()).hexdigest(),
                "PYTHON_IMPORT" if remote else "CHIP_CALLABLE",
                "REMOTE_TASK_DISPATCHER" if remote else "LOCAL_CHIP",
            )

        def add_mpirun_worker_group(self, spec):
            state.spec = spec
            return (7, 13)

        def init(self, **kwargs):
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
            events.append("closed")
            self.memory.clear()

    monkeypatch.setattr(runner, "Worker", Worker)
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


def _i2_compile(root, *, second=False, two_slots=False, reverse=False, dtype="FP32", nbytes=32):
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
        out = self.chip(x, count, data{credit_arg}, out, worker=1)
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


def _i2_binding(program, *, key="activations", slots=None, domain=0):
    binding = getattr(ir, "SharedDomainBinding", None)
    assert callable(binding), "I2 requires the public pypto.ir.SharedDomainBinding API"
    return binding(program=program, domain=domain, key=key, slots={0: "values"} if slots is None else slots)


def test_i2_native_domains_are_immutable_cold_metadata(i2_programs, monkeypatch):
    left, right = i2_programs
    assert hasattr(left, "comm_domains"), "I2 compiled programs must expose immutable comm_domains"
    assert isinstance(left.comm_domains, tuple) and len(left.comm_domains) == 1
    domains = left.comm_domains
    assert domains[0]["ordinal"] == 0 and len(domains[0]["slots"]) == 1
    slot = domains[0]["slots"][0]
    assert (slot["ordinal"], slot["logical_nbytes"], slot["allocated_nbytes"]) == (0, 32, 64)
    assert slot["generated_name"] != right.comm_domains[0]["slots"][0]["generated_name"]
    with pytest.raises(TypeError):
        domains[0]["ordinal"] = 9
    with pytest.raises(TypeError):
        slot["logical_nbytes"] = 64
    metadata = json.loads((left.output_dir / "distributed_meta.json").read_text())
    assert metadata["cluster_domains"]["version"] == 1
    for source in left.output_dir.rglob("*.py"):
        source.write_text("raise AssertionError('cold domain inspection executed source')\n")

    def forbidden(*args, **kwargs):
        pytest.fail("Cold domain restoration must not compile or execute generated Python")

    monkeypatch.setattr(ir, "compile", forbidden)
    monkeypatch.setattr(_artifact_sources, "read_kernel_config", forbidden)
    cold = ir.DistributedCompiledProgram.from_dir(left.output_dir, platform="a2a3")
    assert cold.program is None and cold.comm_domains == domains


@pytest.fixture
def i2_runner_harness(cluster_runner_harness, monkeypatch):
    """Keep native roots, TaskArgs, handle/member classes and registration identities.

    Only process/device transport is recorded. Physical-window bytes here model
    that boundary; actual arithmetic/TLOAD is tested by the modern JIT ST.
    """
    runner, state = cluster_runner_harness
    from simpler.global_comm_domain import GlobalDomainBuffer, GlobalDomainMember  # noqa: PLC0415
    from simpler.task_interface import GlobalCommDomainHandle  # noqa: PLC0415

    base = runner.Worker
    state.allocations, state.observations, state.resets = [], [], []
    state.targets, state.chip_digests = [], []
    state.close_fails = False
    state.mapping_padding = 64

    class Worker(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.drained = True
            self.tearing_down = False
            self.pending_releases = []

        def register(self, target, *, workers=None):
            from simpler.worker import RemoteCallable  # noqa: PLC0415

            handle = super().register(target, workers=workers)
            if isinstance(target, RemoteCallable):
                state.targets.append(target.target)
            else:
                state.chip_digests.append(handle.digest.hex())
            return handle

        def run(self, function, args=None, config=None):
            submitted = False

            def release(handle):
                if self.in_graph:
                    self.pending_releases.append(handle)
                else:
                    assert self.drained or self.tearing_down, "Domain freed before its completion fence"
                    handle._freed = True
                    state.events.append(("domain_free", handle.domain_id))

            def allocate(**kwargs):
                assert self.in_graph and not submitted, "Global allocation must precede the first task"
                members = tuple(
                    GlobalDomainMember(node, local, rank, rank)
                    for rank, (node, local) in enumerate(kwargs["members"])
                )
                assert [(member.node_worker_id, member.local_worker_id) for member in members] == [
                    (7, 0),
                    (7, 1),
                    (13, 0),
                    (13, 1),
                ]
                handle = GlobalCommDomainHandle(
                    name=kwargs["name"],
                    members=members,
                    buffers=tuple(GlobalDomainBuffer(spec.name, spec.nbytes) for spec in kwargs["buffers"]),
                    domain_id=(1 << 63) + len(state.allocations) + 1,
                    generation=1,
                    mapping_size=kwargs["window_size"] + state.mapping_padding,
                    retain_after_run=kwargs.get("retain_after_run", False),
                    _release_fn=release,
                )
                row = {
                    "kwargs": copy.deepcopy(kwargs),
                    "handle": handle,
                    "windows": [bytearray(handle.mapping_size) for _ in members],
                }
                state.allocations.append(row)
                state.events.append(("domain_alloc", handle.domain_id))
                return handle

            def copy_to(handle, domain_rank, data, *, buffer=None, offset=0):
                assert self.in_graph and not submitted, "Reset must precede every task in this run"
                assert 0 < len(data) <= 8 * 1024 * 1024, "R1 global-copy size limit must be respected"
                start = offset + (handle.buffer_range(buffer)[0] if buffer is not None else 0)
                row = next(row for row in state.allocations if row["handle"] is handle)
                assert 0 <= start <= handle.mapping_size - len(data)
                row["windows"][domain_rank][start : start + len(data)] = data
                state.resets.append((handle.domain_id, domain_rank, start, len(data)))

            def submit(handle, incoming, cfg, *, worker):
                nonlocal submitted
                submitted = True
                domain_id = incoming.scalar(incoming.scalar_count() - 1)
                row = next(row for row in state.allocations if row["handle"].domain_id == domain_id)
                assert not row["handle"].released
                state.observations.append((domain_id, tuple(bytes(window) for window in row["windows"])))
                state.tasks.append((handle, incoming, worker))
                state.events.append(("submit", domain_id))

            self.in_graph, self.drained = True, False
            try:
                function(
                    SimpleNamespace(
                        allocate_global_domain=allocate,
                        copy_to_global_domain=copy_to,
                        submit_next_level=submit,
                    ),
                    args,
                    config,
                )
            finally:
                self.in_graph = False
            if state.fail:
                raise RuntimeError("remote DAG failed")
            self.drained = True
            state.events.append("drained")
            for handle in self.pending_releases:
                release(handle)
            self.pending_releases.clear()

        def close(self):
            if state.close_fails:
                raise RuntimeError("hierarchy close failed")
            self.tearing_down = True
            self.drained = True
            state.events.append("tree_drained")
            for row in state.allocations:
                handle = row["handle"]
                if not handle.freed:
                    handle.release()
            super().close()

    monkeypatch.setattr(runner, "Worker", Worker)
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


@pytest.mark.parametrize("mode", ["shared", "distinct_keys", "isolated"])
def test_i2_persistent_programs_share_only_explicit_keys(i2_programs, i2_runner_harness, mode):
    _, state = i2_runner_harness
    left, right = i2_programs
    options = {"extra_compiled": [right], "persistent": True, "reset_persistent_windows": False}
    if mode != "isolated":
        options["shared_domains"] = [
            _i2_binding(left),
            _i2_binding(right, key="other" if mode == "distinct_keys" else "activations"),
        ]
    with left.prepare(**options) as prepared:
        assert len(state.spec.startup_artifacts) == 1, "All programs require one aggregate source package"
        assert all(hook is not None for hook in state.spec.startup_hooks_by_rank)
        assert len(set(state.targets)) == 2, "Same entry names require program-qualified import targets"
        for _ in range(2):
            prepared.run(left, torch.ones(8), 11, torch.zeros(8))
            prepared.run(right, torch.ones(8), 23, torch.zeros(8))
        ids = [row[0] for row in state.observations]
        assert len(state.allocations) == (1 if mode == "shared" else 2)
        assert all(row["handle"].retain_after_run and not row["handle"].freed for row in state.allocations)
        assert (len(set(ids)) == 1) == (mode == "shared")
        assert ids[:4] == ids[4:], "Repeated rounds must retain the same physical domains"
        assert not state.resets
    assert all(row["handle"].freed for row in state.allocations)
    released = [event[1] for event in state.events if isinstance(event, tuple) and event[0] == "domain_free"]
    assert len(released) == len(set(released)) == len(state.allocations)


@pytest.mark.parametrize("reset", [None, False])
def test_i2_reset_covers_every_rank_and_mapping_padding(i2_programs, i2_runner_harness, reset):
    _, state = i2_runner_harness
    left, right = i2_programs
    with left.prepare(
        extra_compiled=[right],
        persistent=True,
        reset_persistent_windows=reset,
        shared_domains=[_i2_binding(left), _i2_binding(right)],
    ) as prepared:
        prepared.run(left, torch.ones(8), 11, torch.zeros(8))
        assert not state.resets, "Freshly allocated windows must not be reset twice"
        row = state.allocations[0]
        for rank, window in enumerate(row["windows"]):
            window[:] = bytes([rank + 1]) * len(window)
        before = tuple(bytes(window) for window in row["windows"])
        prepared.run(right, torch.ones(8), 23, torch.zeros(8))
        observed = state.observations[-1][1]
        expected = before if reset is False else tuple(bytes(len(window)) for window in before)
        assert observed == expected
        if reset is None:
            assert {row[1] for row in state.resets} == set(range(4))
            assert all(not any(window) for window in row["windows"]), "Mapping tail must also be zero"
        else:
            assert not state.resets


def test_i2_large_reset_uses_bounded_global_control_chunks(i2_programs, i2_runner_harness):
    _, state = i2_runner_harness
    left, right = i2_programs
    state.mapping_padding = 8 * 1024 * 1024 + 64
    with left.prepare(
        extra_compiled=[right], persistent=True, shared_domains=[_i2_binding(left), _i2_binding(right)]
    ) as prepared:
        prepared.run(left, torch.ones(8), 11, torch.zeros(8))
        for window in state.allocations[0]["windows"]:
            window[:] = bytes([91]) * len(window)
        prepared.run(right, torch.ones(8), 23, torch.zeros(8))
        for rank in range(4):
            spans = sorted((offset, nbytes) for _, r, offset, nbytes in state.resets if r == rank)
            assert len(spans) >= 2 and spans[0][0] == 0
            end = 0
            for offset, nbytes in spans:
                assert offset == end and 0 < nbytes <= 8 * 1024 * 1024
                end += nbytes
            assert end == state.allocations[0]["handle"].mapping_size


@pytest.mark.parametrize(
    "case",
    ["foreign_program", "duplicate", "domain", "missing_slot", "empty_key", "alias_slots", "nonpersistent"],
)
def test_i2_invalid_sharing_fails_before_worker_initialization(
    i2_programs, i2_runner_harness, case, tmp_path
):
    _, state = i2_runner_harness
    left, right = i2_programs
    bindings = [_i2_binding(left), _i2_binding(right)]
    persistent = True
    if case == "foreign_program":
        bindings[1] = _i2_binding(_i2_compile(tmp_path / "foreign"))
    elif case == "duplicate":
        bindings.append(_i2_binding(left, key="duplicate"))
    elif case == "domain":
        bindings[0] = _i2_binding(left, domain=99)
    elif case == "missing_slot":
        bindings[0] = _i2_binding(left, slots={})
    elif case == "empty_key":
        bindings[0] = _i2_binding(left, key="")
    elif case == "alias_slots":
        left = _i2_compile(tmp_path / "two-slots", two_slots=True)
        bindings[0] = _i2_binding(left, slots={0: "same", 1: "same"})
    else:
        persistent = False
    with pytest.raises((ValueError, TypeError), match=r"(?i)(shar|program|domain|slot|key|persistent)"):
        left.prepare(extra_compiled=[right], persistent=persistent, shared_domains=bindings)
    assert "ready" not in state.events and not state.allocations and not state.tasks


@pytest.mark.parametrize("conflict", ["dtype", "window_size", "topology", "runtime", "profile"])
def test_i2_shared_key_compatibility_is_checked_before_startup(
    i2_programs, i2_runner_harness, conflict, tmp_path
):
    _, state = i2_runner_harness
    left, right = i2_programs
    if conflict == "dtype":
        right = _i2_compile(tmp_path / "different-dtype", second=True, dtype="INT32")
    elif conflict == "window_size":
        right = _i2_compile(tmp_path / "different-size", second=True, nbytes=128)
    else:
        path = right.output_dir / "distributed_meta.json"
        metadata = json.loads(path.read_text())
        config = metadata["cluster_config"]
        if conflict == "topology":
            config["nodes"][0]["devices"].reverse()
        elif conflict == "runtime":
            config["runtime"] = "host_build_graph"
        else:
            config["comm_profile"] = "sim"
        path.write_text(json.dumps(metadata))
        right = ir.DistributedCompiledProgram.from_dir(right.output_dir, platform="a2a3")
    with pytest.raises(
        ValueError, match=r"(?i)(compat|shar|dtype|layout|view|window|topolog|runtime|profile|config)"
    ):
        left.prepare(
            extra_compiled=[right], persistent=True, shared_domains=[_i2_binding(left), _i2_binding(right)]
        )
    assert "ready" not in state.events and not state.allocations


def test_i2_explicit_slot_permutation_uses_one_canonical_layout(i2_runner_harness, tmp_path):
    _, state = i2_runner_harness
    left = _i2_compile(tmp_path / "two-left", two_slots=True)
    right = _i2_compile(tmp_path / "two-right", second=True, two_slots=True, reverse=True)
    assert hasattr(left, "comm_domains"), "I2 requires cold domain/slot introspection"

    def slots(program):
        return {
            slot["ordinal"]: "values" if slot["logical_nbytes"] == 32 else "credits"
            for slot in program.comm_domains[0]["slots"]
        }

    left_map, right_map = slots(left), slots(right)
    assert left_map != right_map, "The native fixtures must exercise different allocation ordering"
    with left.prepare(
        extra_compiled=[right],
        persistent=True,
        reset_persistent_windows=False,
        shared_domains=[_i2_binding(left, slots=left_map), _i2_binding(right, slots=right_map)],
    ) as prepared:
        prepared.run(left, torch.ones(8), 11, torch.zeros(8))
        prepared.run(right, torch.ones(8), 23, torch.zeros(8))
        assert len(state.allocations) == 1
        handle = state.allocations[0]["handle"]
        assert [buffer.name for buffer in handle.buffers] == ["credits", "values"]
        assert handle.buffer_range("credits") == (0, 64)
        assert handle.buffer_range("values") == (64, 64)


def test_i2_prepare_owns_slot_map_snapshot(i2_programs, i2_runner_harness):
    _, state = i2_runner_harness
    left, right = i2_programs
    original = {0: "values"}
    bindings = [_i2_binding(left, slots=original), _i2_binding(right)]
    with left.prepare(
        extra_compiled=[right], persistent=True, reset_persistent_windows=False, shared_domains=bindings
    ) as prepared:
        original[0] = "changed-after-prepare"
        bindings.clear()
        prepared.run(left, torch.ones(8), 11, torch.zeros(8))
        prepared.run(right, torch.ones(8), 23, torch.zeros(8))
        assert len(state.allocations) == 1
        assert [buffer.name for buffer in state.allocations[0]["handle"].buffers] == ["values"]


def test_i2_uncertain_close_retains_domain_and_io_journals(i2_programs, i2_runner_harness):
    _, state = i2_runner_harness
    left, right = i2_programs
    prepared = left.prepare(
        extra_compiled=[right],
        persistent=True,
        reset_persistent_windows=False,
        shared_domains=[_i2_binding(left), _i2_binding(right)],
    )
    try:
        prepared.run(left, torch.ones(8), 11, torch.zeros(8))
        retained = state.allocations[0]["handle"]
        failure_cursor = len(state.events)
        state.fail = state.close_fails = True
        with pytest.raises(RuntimeError, match="hierarchy close failed"):
            prepared.run(right, torch.ones(8), 23, torch.zeros(8))
        assert not retained.released and not retained.freed
        io, keep = prepared._active_io, tuple(prepared._keep)
        assert io is not None and io.storages and keep
        before = list(state.events)
        with pytest.raises(RuntimeError, match=r"(?i)(pending|previous|clos|fail)"):
            prepared.run(left, torch.ones(8), 31, torch.zeros(8))
        assert state.events == before and prepared._active_io is io and tuple(prepared._keep) == keep
        assert not any(
            isinstance(event, tuple) and event[0] == "copyback" for event in state.events[failure_cursor:]
        )
    finally:
        state.close_fails = False
        prepared.close()
    assert retained.freed and prepared._active_io is None and not prepared._keep
    assert not state.worker.memory


def test_i2_aggregate_is_deterministic_plain_ustar_and_program_qualified(i2_programs, i2_runner_harness):
    _, state = i2_runner_harness
    left, right = i2_programs
    payloads = []
    for _ in range(2):
        with left.prepare(
            extra_compiled=[right], persistent=True, shared_domains=[_i2_binding(left), _i2_binding(right)]
        ):
            assert len(state.spec.startup_artifacts) == 1
            artifact = state.spec.startup_artifacts[0]
            payload = Path(artifact.archive_path).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == artifact.sha256
            payloads.append(payload)
            raw = gzip.decompress(payload)
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                names = archive.getnames()
                assert names == sorted(set(names))
                assert "session_manifest.json" in names
                for program in ("p0", "p1"):
                    base = f"programs/{program}"
                    assert base + "/distributed_meta.json" in names
                    assert base + "/next_levels/node/orchestration/host_orch.py" in names
                    shim = archive.extractfile(base + "/orchestration/_distributed_support.py").read()
                    assert b"class _EmbeddedNodeState" not in shim
                implementations = []
                for member in archive.getmembers():
                    assert member.isfile() and member.uid == member.gid == member.mtime == 0
                    assert raw[member.offset + 257 : member.offset + 265] == b"ustar\x0000"
                    data = archive.extractfile(member).read()
                    assert artifact.sha256.encode() not in data
                    if b"class _EmbeddedNodeState" in data:
                        implementations.append(member.name)
                assert implementations == ["orchestration/_distributed_support.py"]
    assert payloads[0] == payloads[1], "Session sources must not embed temporary paths or session identities"


@pytest.mark.parametrize("budget", ["members", "unpacked"])
def test_i2_aggregate_caps_apply_to_whole_session(i2_programs, i2_runner_harness, monkeypatch, budget):
    _, state = i2_runner_harness
    left, right = i2_programs
    _, single_payload = _artifact_sources.build_cluster_source_bundle(left.output_dir)
    with tarfile.open(fileobj=io.BytesIO(single_payload), mode="r:gz") as archive:
        members = archive.getmembers()
        if budget == "members":
            monkeypatch.setattr(_artifact_sources, "_CLUSTER_MAX_MEMBERS", len(members) + 2)
        else:
            monkeypatch.setattr(
                _artifact_sources, "_CLUSTER_MAX_UNPACKED_BYTES", sum(member.size for member in members) + 128
            )
    with pytest.raises(ValueError, match=r"(?i)(member|size|limit|unpacked)"):
        left.prepare(
            extra_compiled=[right], persistent=True, shared_domains=[_i2_binding(left), _i2_binding(right)]
        )
    assert "ready" not in state.events


_I2_SESSION_PROOF = (
    _RUNTIME_IMPORT_PROOF.split("namespace, digest, archive_path, cache_root", 1)[0]
    + r"""
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import torch
import simpler.buffer as buffer_api
from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id
from simpler.startup_artifacts import (
    StartupArtifactSpec, read_startup_artifacts, install_startup_artifacts, activate_startup_artifacts,
)
from simpler.startup_preparation import NodeStartupContext
from simpler.task_interface import (
    CallConfig, ChipDomainContext, DataType, GlobalCommDomainView, GlobalDomainMember,
    TaskArgs, TensorArgType,
)
from simpler.worker import Worker

archive_path, description_path, cache_root = sys.argv[1:]
description = json.loads(Path(description_path).read_text())
artifact = description['artifact']
namespace = artifact['module_prefix']
artifact['archive_path'] = archive_path
installed = install_startup_artifacts(read_startup_artifacts([artifact]), cache_root)
activate_startup_artifacts(installed)
hook = description['hook']
module, qualname = hook['import_path'].split(':')
prepare = getattr(importlib.import_module(module), qualname)
inner = SimpleNamespace(_config={'device_ids': [12, 15]}, register=lambda function: object())
bindings = prepare(NodeStartupContext(
    rank=0, artifact_roots={artifact['sha256']: installed[0].root}, inner_worker=inner,
    chip_handles={digest: object() for digest in description['chip_digests']},
    parameters=hook['parameters'],
))
entries = []
for target in description['targets']:
    module, qualname = target.split(':')
    entries.append(getattr(importlib.import_module(module), qualname))
assert len(entries) == 2 and entries[0] is not entries[1]
support = importlib.import_module(namespace + '.orchestration._distributed_support')
for slot in ('p0', 'p1'):
    shim = importlib.import_module(namespace + '.programs.' + slot + '.orchestration._distributed_support')
    assert shim._EmbeddedNodeState is support._EmbeddedNodeState
    assert shim._ChipCallables is support._ChipCallables
states = [entry._pypto_embedded_state for entry in entries]
node = states[0].node_state
assert node is states[1].node_state and node.local_worker_ids == (0, 1)
assert isinstance(states[0].callables, support._ChipCallables)
assert isinstance(states[1].callables, support._ChipCallables)
capacity = len(node._tokens)
assert capacity == sum(row[1] for entry in entries
                       for row in entry._pypto_embedded_layout['prefork_requirements']
                       if row[0] == 'inherited_domain_tokens')
assert all(token.is_shared() for bundle in node._tokens for token in bundle)

owner = mint_owner_instance_id()
backings = [create_host_shared_buffer(64, owner, index + 1) for index in range(4)]
worker = Worker(level=3, num_sub_workers=0)
try:
    worker.init()
    x, out = [backing.tensor(shapes=(8,), dtype=DataType.FLOAT32) for backing in backings[:2]]
    physical = description['physical']
    assert physical['buffers'] == [['values', 64]]
    members = tuple(GlobalDomainMember(row[0], row[1], row[2], row[3]) for row in physical['members'])
    view = None
    calls = []
    def resolve(domain_id):
        assert domain_id == view.domain_id
        return view
    def submit(handle, incoming, config, *, worker):
        assert worker == 1
        calls.append(incoming)
    orch = SimpleNamespace(_worker=worker, get_global_domain=resolve, submit_next_level=submit)
    def incoming():
        args = TaskArgs()
        args.add_tensor(x, TensorArgType.INPUT)
        args.add_tensor(out, TensorArgType.OUTPUT_EXISTING)
        args.add_scalar(ctypes.c_int32(7))
        args.add_scalar(ctypes.c_uint64(view.domain_id))
        return args
    def forbidden(*args, **kwargs):
        raise AssertionError('I2 allocated shared storage after bootstrap')
    with mock.patch.object(torch, 'zeros', forbidden), mock.patch.object(torch, 'empty', forbidden), \
         mock.patch.object(torch.Tensor, 'share_memory_', forbidden), \
         mock.patch.object(buffer_api, 'create_host_shared_buffer', forbidden):
        for generation in range(1, capacity + 4):
            contexts = {local: ChipDomainContext(
                name=physical['name'], domain_rank=local, domain_size=4,
                device_ctx=70000 + local, local_window_base=int(backings[2 + local].base),
                actual_window_size=64, buffers={'values': backings[2 + local]},
            ) for local in (0, 1)}
            view = GlobalCommDomainView(name=physical['name'], members=members, contexts=contexts,
                domain_id=physical['domain_id'] + generation, generation=generation, mapping_size=64)
            view._committed = True
            for broken in ('missing_slot', 'misaligned_base', 'wrong_size', 'uncommitted'):
                context = view.contexts[1]
                before = dict(node._bindings)
                original_base, original_size = context.local_window_base, context.actual_window_size
                removed = context.buffers.pop('values') if broken == 'missing_slot' else None
                if broken == 'misaligned_base':
                    context.local_window_base += 1
                elif broken == 'wrong_size':
                    context.actual_window_size = 32
                elif broken == 'uncommitted':
                    view._committed = False
                calls.clear()
                try:
                    entries[0](orch, incoming(), CallConfig())
                except (ValueError, RuntimeError, KeyError):
                    pass
                else:
                    raise AssertionError('Invalid physical view reached CHIP: ' + broken)
                finally:
                    context.local_window_base, context.actual_window_size = original_base, original_size
                    if removed is not None:
                        context.buffers['values'] = removed
                    view._committed = True
                assert not calls and node._bindings == before and states[0]._frame is None
            shared_token = None
            for entry, state in zip(entries, states):
                calls.clear()
                entry(orch, incoming(), CallConfig())
                assert len(calls) == 1 and state._frame is not None
                outgoing = calls[0]
                assert outgoing.tensor_count() == 4 and outgoing.scalar_count() == 2
                expected = backings[3].tensor(shapes=(8,), dtype=DataType.FLOAT32)
                assert outgoing.tensor(1) == expected
                assert outgoing.scalar(1) == view[1].device_ctx == 70001
                token = outgoing.tensor(3)
                assert outgoing.tag(3) == TensorArgType.INOUT
                if shared_token is not None:
                    assert token == shared_token, 'Two programs lost the canonical ordering token'
                shared_token = token
                bindings.task_completed()
                assert state._frame is None
                assert (view.domain_id, view.generation) in node._bindings
            assert len(node._bindings) <= capacity
            view._committed = False
        assert len(node._tokens) == capacity
    assert not guard.attempts
    print('I2_CANONICAL_SESSION_PASS')
finally:
    worker.close()
    bindings.close()
    for backing in backings:
        backing.close()
"""
)


def test_i2_installed_native_entries_share_actual_views_and_prefork_pool(
    i2_programs, i2_runner_harness, tmp_path
):
    runtime_python = os.environ.get("L4_RUNTIME_ONLY_PYTHON")
    runtime_pythonpath = os.environ.get("L4_RUNTIME_ONLY_PYTHONPATH")
    if not runtime_python or not runtime_pythonpath:
        pytest.skip("Set L4_RUNTIME_ONLY_PYTHON and L4_RUNTIME_ONLY_PYTHONPATH")
    _, state = i2_runner_harness
    left, right = i2_programs
    with left.prepare(
        extra_compiled=[right],
        persistent=True,
        reset_persistent_windows=False,
        shared_domains=[_i2_binding(left), _i2_binding(right)],
    ) as prepared:
        prepared.run(left, torch.ones(8), 11, torch.zeros(8))
        artifact = state.spec.startup_artifacts[0]
        archive = tmp_path / "aggregate-proof.tar.gz"
        archive.write_bytes(Path(artifact.archive_path).read_bytes())
        handle = state.allocations[0]["handle"]
        description = {
            "artifact": artifact.to_manifest(),
            "hook": state.spec.startup_hooks_by_rank[0].to_manifest(),
            "targets": state.targets,
            "chip_digests": state.chip_digests,
            "physical": {
                "name": handle.name,
                "domain_id": handle.domain_id,
                "buffers": [[buffer.name, buffer.nbytes] for buffer in handle.buffers],
                "members": [
                    [
                        member.node_worker_id,
                        member.local_worker_id,
                        member.global_device_rank,
                        member.domain_rank,
                    ]
                    for member in handle.members
                ],
            },
        }
    description_path = tmp_path / "aggregate-proof.json"
    description_path.write_text(json.dumps(description))
    result = subprocess.run(
        [
            runtime_python,
            "-c",
            _I2_SESSION_PROOF,
            str(archive),
            str(description_path),
            str(tmp_path / "installed-i2"),
        ],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": runtime_pythonpath, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "I2_CANONICAL_SESSION_PASS" in result.stdout


def test_i2_nonpersistent_multi_program_keeps_per_run_domain_lifetimes(i2_programs, i2_runner_harness):
    _, state = i2_runner_harness
    left, right = i2_programs
    with left.prepare(extra_compiled=[right]) as prepared:
        for program in (left, right, left, right):
            prepared.run(program, torch.ones(8), 11, torch.zeros(8))
        assert len(state.allocations) == 4 and not state.resets
        assert all(not row["handle"].retain_after_run and row["handle"].freed for row in state.allocations)
        assert len({row[0] for row in state.observations}) == 4


@pytest.mark.parametrize("omission", ["one", "all", "domain"])
def test_i2_cold_domain_views_require_complete_native_coverage(tmp_path, omission):
    compiled = _i2_compile(tmp_path / "complete-native-views", two_slots=True)
    path = compiled.output_dir / "distributed_meta.json"
    metadata = json.loads(path.read_text())
    domain = metadata["cluster_domains"]["domains"][0]
    assert len(domain["slots"]) == len(domain["entry_windows"]) == 2
    native_views = metadata["l3_entries"]["entries"][0]["layout"]["window_slots"]
    assert len(native_views) == 2 and {row[5] for row in native_views} == {"fp32", "int32"}
    if omission == "domain":
        metadata["cluster_domains"]["domains"] = []
    else:
        domain["entry_windows"] = domain["entry_windows"][:1] if omission == "one" else []
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=r"(?i)cluster_domains.*(view|window|coverage)"):
        ir.DistributedCompiledProgram.from_dir(compiled.output_dir, platform="a2a3")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
