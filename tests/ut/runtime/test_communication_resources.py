# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resource contracts bind explicit promises; static warmup never runs a Host body."""

import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pypto.runtime.communication_resources import (
    RESOURCE_CONTRACT_FILE,
    entry_has_control_effects,
    load_domain_plan,
    write_resource_contract,
)

HOST_SOURCE = """def entry(orch, _args, config, *, tensors, callables, sub_ids, _keep,
          world_size, _domain_provider=None):
    with (_domain_provider or orch.allocate_domain)(
        name="comm", workers=[*range(world_size)], window_size=64 * 2,
        buffers=[CommBufferSpec(name="window", dtype="opaque", count=128, nbytes=128)],
    ) as domain:
        for rank in range(world_size):
            args = TaskArgs()
            args.add_scalar(rank)
            args.add_scalar(domain[rank].device_ctx)
            _submit_chip(orch, callables["chip"], args, config, rank)
"""


@pytest.fixture
def artifact(tmp_path):
    root = tmp_path / "artifact"
    host = root / "orchestration" / "host_orch.py"
    host.parent.mkdir(parents=True)
    host.write_text(HOST_SOURCE)
    native = root / "next_levels" / "chip" / "orchestration" / "chip.so"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"final-native-contents")
    return root, host, SimpleNamespace(__name__="entry")


def issue(root: Path, dependencies=()):
    write_resource_contract(root, complete_resources=True, external_dependencies=dependencies, world_size=2)


def test_contract_requires_explicit_promise_and_final_binary(artifact):
    root, host, entry = artifact
    assert load_domain_plan(entry, host, 2) is None
    with pytest.raises(TypeError):
        write_resource_contract(root)
    with pytest.raises(ValueError, match="explicit"):
        write_resource_contract(root, complete_resources=False, external_dependencies=(), world_size=2)
    (root / "next_levels" / "chip" / "orchestration" / "chip.so").unlink()
    with pytest.raises(ValueError, match="final native"):
        issue(root)


def test_static_plan_never_executes_host_graph(artifact, monkeypatch):
    root, host, entry = artifact
    host.write_text(HOST_SOURCE + "    raise AssertionError('must not execute')\n")
    issue(root)
    plan = load_domain_plan(entry, host, 2)
    assert plan is not None
    assert plan.declarations[0].workers == (0, 1)
    captures = []

    @contextmanager
    def provider(**kwargs):
        captures.append(kwargs)
        yield object()

    interface = ModuleType("simpler.task_interface")
    setattr(interface, "CommBufferSpec", SimpleNamespace)
    monkeypatch.setitem(sys.modules, "simpler.task_interface", interface)
    plan.warm(provider)
    assert len(captures) == 1
    assert captures[0]["name"] == "comm"
    assert captures[0]["buffers"][0].nbytes == 128


@pytest.mark.parametrize(
    "relative",
    [
        "orchestration/host_orch.py",
        "next_levels/chip/kernel_config.py",
        "next_levels/chip/helper.h",
        "next_levels/chip/orchestration/chip.so",
        "distributed_meta.json",
    ],
)
def test_contract_rejects_changed_artifact_or_configuration(artifact, relative):
    root, host, entry = artifact
    changed = root / relative
    if not changed.exists():
        changed.write_text("initial")
    issue(root)
    assert load_domain_plan(entry, host, 2) is not None
    changed.write_bytes(changed.read_bytes() + b"\nchanged")
    assert load_domain_plan(entry, host, 2) is None


def test_contract_tracks_external_dependency_closure(artifact, tmp_path):
    root, host, entry = artifact
    dependency = tmp_path / "helper.h"
    with pytest.raises(FileNotFoundError):
        issue(root, (dependency,))
    dependency.write_text("known implementation")
    issue(root, (dependency,))
    assert load_domain_plan(entry, host, 2) is not None
    dependency.write_text("changed implementation")
    assert load_domain_plan(entry, host, 2) is None


def test_contract_rejects_missing_manifest_changed_inventory_and_world(artifact):
    root, host, entry = artifact
    issue(root)
    assert load_domain_plan(entry, host, 3) is None
    extra = root / "next_levels" / "new_helper.h"
    extra.write_text("new code")
    assert load_domain_plan(entry, host, 2) is None
    extra.unlink()
    assert load_domain_plan(entry, host, 2) is not None
    (host.parent / RESOURCE_CONTRACT_FILE).unlink()
    assert load_domain_plan(entry, host, 2) is None


@pytest.mark.parametrize(
    "old, new",
    [
        ("window_size=64 * 2", 'window_size=tensors["extent"]'),
        ("args.add_scalar(rank)", 'args.add_scalar(tensors["pointer"])'),
        ("window_size=64 * 2", "window_size=64 * 2, workspace_engines=('urma',)"),
        ("window_size=64 * 2", "window_size=orch.query_dynamic_size()"),
    ],
)
def test_unknown_resource_provenance_is_not_certified(artifact, old, new):
    root, host, entry = artifact
    issue(root)
    host.write_text(HOST_SOURCE.replace(old, new))
    with pytest.raises(ValueError, match="supported static"):
        issue(root)
    assert load_domain_plan(entry, host, 2) is None


def test_control_free_entry_is_distinguished_from_unknown_control_helpers(artifact):
    _root, host, entry = artifact
    assert entry_has_control_effects(entry, host)
    host.write_text(
        "def entry(orch):\n    args = TaskArgs()\n    args.add_scalar(1)\n    _submit_chip(orch, args)\n"
    )
    assert not entry_has_control_effects(entry, host)
    host.write_text("def entry(orch):\n    unknown_helper(orch)\n")
    assert entry_has_control_effects(entry, host)
