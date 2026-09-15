# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Kernel descriptors are checked against both metadata and the pinned SDK."""

import json
import shutil
import struct
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from pypto._artifact_contract import ArtifactExecutionMode, ExecutionCapabilities
from pypto._identity import ToolchainIdentity, digest_record
from pypto._kernel_abi import (
    MAX_KERNEL_RANK,
    MAX_KERNEL_SCALARS,
    MAX_KERNEL_TENSORS,
    SCALAR_FORMATS,
    SIMPLER_KERNEL_REVISION,
    TENSOR_DIRECTION_TAGS,
    TENSOR_DTYPE_TAGS,
    KernelABI,
    KernelParameter,
)
from pypto.ir.compiled_program import CompiledProgram, load_kernel_metadata, write_kernel_metadata
from pypto.ir.param_info import ParamInfo, bind_kernel_args, kernel_abi_from_params
from pypto.jit._artifact_manifest import ArtifactKey, ArtifactSpec, ArtifactState, BuildKind
from pypto.jit.artifact_cache import ArtifactStore, LookupStatus
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime._artifact_runtime import ArtifactRuntime, restore_kernel_metadata
from pypto.runtime._prebuilt import ready_spec


@pytest.fixture
def params():
    return [
        ParamInfo("x", ParamDirection.In, [8], DataType.FP32),
        ParamInfo("scale", ParamDirection.In, None, DataType.FP32),
        ParamInfo("out", ParamDirection.Out, [8], DataType.FP32),
        ParamInfo("step", ParamDirection.In, None, DataType.INT32),
        ParamInfo("cache", ParamDirection.InOut, [8], DataType.FP32),
    ]


@pytest.fixture
def abi(params):
    return kernel_abi_from_params(
        params, platform="a2a3", runtime="tensormap_and_ringbuffer", return_aliases=(2, 4, 2)
    )


def test_interleaved_pools_preserve_values_and_aliases(params, abi):
    x, out = object(), object()
    for scale, step in ((1.25, 0), (-3.5, 9)):
        tensors, scalars, returns = bind_kernel_args([x, scale, out, step, x], params, abi)
        assert tensors == [x, out, x]
        assert scalars == [scale, step]
        assert returns == [out, x, out]
        assert returns[0] is returns[2]
    record = abi.record()
    assert [p.get("tensor_index") for p in record["params"]] == [0, None, 1, None, 2]
    assert [p.get("scalar_index") for p in record["params"]] == [None, 0, None, 1, None]
    assert KernelABI.from_record(json.loads(json.dumps(record))) == abi
    assert replace(abi, return_aliases=(0,)).return_aliases == (0,)
    with pytest.raises(TypeError, match="including all Out/InOut"):
        bind_kernel_args([x, 1.25, out, 0], params, abi)
    changed = [replace(params[0], dtype=DataType.INT32), *params[1:]]
    with pytest.raises(ValueError, match="does not match"):
        bind_kernel_args([x, 1.25, out, 0, x], changed, abi)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", 2),
        ("schema", True),
        ("schema", 1.0),
        ("simpler_revision", "0" * 40),
        ("argument_abi", "TaskArgs"),
        ("scalar_storage", "cast_to_uint64"),
        ("tensor_layout", "contiguous"),
        ("return_aliases", [1]),
        ("return_aliases", [True]),
        ("return_aliases", [99]),
        ("runtime", "other"),
        ("platform", "a2a3sim"),
    ],
)
def test_bad_descriptor_is_rejected(abi, field, value):
    record = abi.record()
    record[field] = value
    with pytest.raises(ValueError, match="kernel ABI"):
        KernelABI.from_record(record)


@pytest.mark.parametrize(
    "index,field,value",
    [
        (0, "dtype_tag", 1),
        (2, "direction_tag", 1),
        (0, "tensor_index", 1),
        (2, "tensor_index", 0),
        (1, "scalar_index", 1),
        (1, "encoding", "Q"),
        (0, "shape", [True]),
        (0, "shape", [0]),
        (0, "shape", [2**32]),
        (0, "dtype", "fp64"),
        (1, "dtype", "fp16"),
        (1, "direction", "Out"),
        (0, "name", "out"),
    ],
)
def test_bad_parameter_mapping_is_rejected(abi, index, field, value):
    record = abi.record()
    record["params"][index][field] = value
    with pytest.raises(ValueError):
        KernelABI.from_record(record)


def test_descriptor_is_immutable_and_limits_are_checked(abi):
    record = abi.record()
    record["params"][0]["shape"].clear()
    record["return_aliases"].clear()
    assert abi.parameters[0].shape == (8,)
    assert abi.return_aliases == (2, 4, 2)
    for shape in (None, (1,)):
        limit = MAX_KERNEL_SCALARS if shape is None else MAX_KERNEL_TENSORS
        params = tuple(KernelParameter(f"p{i}", "int32", "In", shape) for i in range(limit))
        assert len(replace(abi, parameters=params, return_aliases=()).parameters) == limit
        with pytest.raises(ValueError, match="capacity"):
            replace(abi, parameters=(*params, replace(params[0], name="extra")), return_aliases=())
    with pytest.raises(ValueError, match="rank"):
        KernelParameter("x", "fp32", "In", (1,) * 6)
    assert KernelParameter("x", "fp32", "In", (-1, 8)).shape == (-1, 8)


def test_metadata_round_trip_is_device_free_and_program_rejects_kernel(tmp_path, params, abi):
    write_kernel_metadata(tmp_path, params, abi)
    restored = load_kernel_metadata(tmp_path, abi)
    assert restored["kernel_abi"] == abi
    assert [p.name for p in restored["param_infos"]] == [p.name for p in params]
    with pytest.raises(ValueError, match="requires 'program'"):
        CompiledProgram.from_dir(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        load_kernel_metadata(tmp_path, replace(abi, runtime="host_build_graph"))


@pytest.mark.parametrize("change", ["missing", "params", "returns", "platform", "backend", "old_schema"])
def test_sidecar_cannot_disagree_with_descriptor(tmp_path, params, abi, change):
    write_kernel_metadata(tmp_path, params, abi)
    path = tmp_path / "compiled_meta.json"
    meta = json.loads(path.read_text())
    if change == "missing":
        del meta["kernel_abi"]
    elif change == "params":
        meta["params"][0]["name"] = "different"
    elif change == "returns":
        meta["num_return_types"] = 0
    elif change == "platform":
        meta["platform"] = "a5"
    elif change == "backend":
        meta["backend_type"] = "Ascend950"
    else:
        meta["schema"] -= 1
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="recompile"):
        load_kernel_metadata(tmp_path, abi)


def test_each_orchestration_has_its_own_signature(tmp_path, params, abi):
    write_kernel_metadata(tmp_path, params, abi)
    child = tmp_path / "next_levels" / "child"
    child.mkdir(parents=True)
    child_params = [params[0], params[2]]
    child_abi = kernel_abi_from_params(
        child_params, platform=abi.platform, runtime=abi.runtime, return_aliases=(1,)
    )
    write_kernel_metadata(child, child_params, child_abi)
    assert load_kernel_metadata(tmp_path, abi)["kernel_abi"] == abi
    assert load_kernel_metadata(child, child_abi)["kernel_abi"] == child_abi
    with pytest.raises(ValueError, match="does not match"):
        load_kernel_metadata(child, abi)


def test_stage_contract_identity_and_restore(tmp_path, params, abi):
    # Use a deterministic complete test toolchain, without probing the installed runtime.
    digest = "a" * 64
    identity = ToolchainIdentity(digest, digest, digest, digest, digest)
    key = ArtifactKey(identity, digest_record("source"), digest_record("specialization"))
    spec = ArtifactSpec(
        ArtifactState.GENERATED,
        BuildKind.SINGLE_CHIP,
        ("compiled_meta.json", "kernel_config.py"),
        ExecutionCapabilities((ArtifactExecutionMode.KERNEL,)),
        abi,
    )
    store = ArtifactStore(tmp_path / "cache", private_root=tmp_path / "private")

    def builder(directory):
        write_kernel_metadata(directory, params, abi)
        (directory / "kernel_config.py").write_text("KERNELS = []\n")

    generated = store.get_or_build(key, spec, builder).handle
    assert generated is not None
    assert restore_kernel_metadata(generated, abi)["kernel_abi"] == abi
    with pytest.raises(ValueError, match="requires 'program'"):
        ArtifactRuntime(store, generated, "a2a3", tmp_path / "run")
    for other in (
        replace(abi, platform="a5"),
        replace(abi, runtime="host_build_graph"),
        replace(abi, return_aliases=(2,)),
    ):
        assert store.lookup(key, replace(spec, kernel_abi=other)).status is LookupStatus.MISS
    ready = replace(spec, state=ArtifactState.BINARY_READY)
    handle = store.get_or_build(key, ready, builder).handle
    assert handle is not None
    assert restore_kernel_metadata(handle, abi)["kernel_abi"] == abi
    manifest = handle.directory / "artifact_manifest.json"
    record = json.loads(manifest.read_text())
    record["kernel_abi"]["runtime"] = "host_build_graph"
    manifest.write_text(json.dumps(record))
    assert store.lookup(key, ready).status is LookupStatus.INVALID


def test_ready_spec_preserves_kernel_abi(tmp_path, params, abi):
    write_kernel_metadata(tmp_path, params, abi)
    # ready_spec inventories config paths without compiling or loading them.
    (tmp_path / "kernel_config.py").write_text(
        f"ORCHESTRATION = dict(source={str(tmp_path / 'orch.cpp')!r}, function_name='entry')\nKERNELS = []\n"
    )
    spec = ArtifactSpec(
        ArtifactState.GENERATED,
        BuildKind.SINGLE_CHIP,
        ("compiled_meta.json", "kernel_config.py"),
        ExecutionCapabilities((ArtifactExecutionMode.KERNEL,)),
        abi,
    )
    assert ready_spec(tmp_path, spec).kernel_abi == abi


def test_contract_matches_pinned_simpler_headers(tmp_path):
    compiler = shutil.which("c++")
    assert compiler is not None, "The project build toolchain is required for ABI conformance"
    root = Path(__file__).resolve().parents[3]
    runtime = root / "runtime"
    revision = subprocess.run(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert revision == SIMPLER_KERNEL_REVISION
    native_names = [
        "FLOAT32",
        "FLOAT16",
        "INT32",
        "INT16",
        "INT8",
        "UINT8",
        "BFLOAT16",
        "INT64",
        "UINT64",
        "UINT16",
        "UINT32",
        "BOOL",
        "FP8E4M3FN",
        "FP8E8M0",
        "FP4E2M1",
    ]
    assertions = [
        f"static_assert(int(DataType::{name}) == {tag});"
        for name, tag in zip(native_names, TENSOR_DTYPE_TAGS.values())
    ]
    source = "\n".join(
        [
            '#include "task_args.h"',
            "#include <iostream>",
            *assertions,
            *[
                f"static_assert(int(ArgDirection::{name.upper()}) == {tag});"
                for name, tag in TENSOR_DIRECTION_TAGS.items()
            ],
            f"static_assert(int(DataType::DATA_TYPE_NUM) == {len(TENSOR_DTYPE_TAGS)});",
            f"static_assert(MAX_TENSOR_DIMS == {MAX_KERNEL_RANK});",
            f"static_assert(CHIP_MAX_TENSOR_ARGS == {MAX_KERNEL_TENSORS});",
            f"static_assert(CHIP_MAX_SCALAR_ARGS == {MAX_KERNEL_SCALARS});",
            "static_assert(sizeof(ChipTensor) == 72);",
            "static_assert(offsetof(ChipStorageTaskArgs, scalars_) == 256 * 72);",
            "int main() { std::cout << to_u64(float(-3.5)) << ' ' << to_u64(int32_t(-7)); }",
        ]
    )
    binary = tmp_path / "abi_probe"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(runtime / "src/common/task_interface"),
            "-I",
            str(runtime / "src/common/platform/include"),
            "-x",
            "c++",
            "-",
            "-o",
            str(binary),
        ],
        check=False,
        input=source,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    actual = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout.split()
    expected = [
        int.from_bytes(struct.pack("<" + SCALAR_FORMATS[dtype], value), "little")
        for dtype, value in (("fp32", -3.5), ("int32", -7))
    ]
    assert [int(value) for value in actual] == expected


@pytest.mark.parametrize("case", ["missing", "program", "shared", "distributed"])
def test_spec_rejects_incompatible_execution_contracts(abi, case):
    modes = (ArtifactExecutionMode.KERNEL,)
    kind = BuildKind.SINGLE_CHIP
    descriptor = abi
    if case == "missing":
        descriptor = None
    elif case == "program":
        modes = (ArtifactExecutionMode.PROGRAM,)
    elif case == "shared":
        modes = (ArtifactExecutionMode.PROGRAM, ArtifactExecutionMode.KERNEL)
    else:
        kind = BuildKind.DISTRIBUTED
    with pytest.raises(ValueError):
        ArtifactSpec(
            ArtifactState.GENERATED, kind, ("compiled_meta.json",), ExecutionCapabilities(modes), descriptor
        )


def test_store_rejects_manifest_sidecar_abi_disagreement(tmp_path, params, abi):
    digest = "a" * 64
    key = ArtifactKey(ToolchainIdentity(digest, digest, digest, digest, digest), digest, digest)
    spec = ArtifactSpec(
        ArtifactState.GENERATED,
        BuildKind.SINGLE_CHIP,
        ("compiled_meta.json",),
        ExecutionCapabilities((ArtifactExecutionMode.KERNEL,)),
        abi,
    )
    store = ArtifactStore(tmp_path / "cache", private_root=tmp_path / "private")
    other = replace(abi, runtime="host_build_graph")
    handle = store.get_or_build(
        key, spec, lambda directory: write_kernel_metadata(directory, params, other)
    ).handle
    assert handle is not None
    with pytest.raises(ValueError, match="does not match"):
        restore_kernel_metadata(handle, abi)
