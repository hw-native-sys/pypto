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
import json

import pypto.language as pl
import pytest
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
