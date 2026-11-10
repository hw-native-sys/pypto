# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared artifact metadata and L2 contract boundaries, without device execution."""

import json
import struct
import subprocess
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from pypto._artifact_contract import ArtifactExecutionMode, ExecutionCapabilities
from pypto.backend import BackendType
from pypto.ir._artifact_metadata import ArtifactMetadata
from pypto.ir._compile_contract import validate_lowered_signature
from pypto.ir.compiled_artifact import CompiledArtifact
from pypto.ir.compiled_program import CompiledProgram
from pypto.ir.param_info import (
    MAX_KERNEL_SCALARS,
    MAX_KERNEL_TENSORS,
    ParamInfo,
    validate_l2_capacity,
    validate_parameter_names,
    validate_return_aliases,
)
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection


@pytest.fixture
def params():
    return [
        ParamInfo("x", ParamDirection.In, [8], DataType.FP32),
        ParamInfo("scale", ParamDirection.In, None, DataType.FP32),
        ParamInfo("out", ParamDirection.Out, [8], DataType.FP32),
        ParamInfo("step", ParamDirection.In, None, DataType.INT32),
        ParamInfo("cache", ParamDirection.InOut, [8], DataType.FP32),
    ]


def metadata(params, aliases=(2, 4, 2), mode=ArtifactExecutionMode.KERNEL):
    return ArtifactMetadata(
        tuple(params),
        tuple(params),
        len(aliases),
        "a2a3sim",
        BackendType.Ascend910,
        aliases,
        ExecutionCapabilities((mode,)),
    )


def test_shared_signature_snapshot_and_duplicate_alias_round_trip(params, tmp_path):
    meta = metadata(params)
    meta.write(tmp_path)
    params[0] = replace(params[0], shape=[9])
    assert ArtifactMetadata.read(tmp_path) == meta
    assert meta.parameters[0].shape == (8,)
    assert meta.return_aliases == (2, 4, 2)
    with pytest.raises(FrozenInstanceError):
        setattr(meta, "return_aliases", ())
    with pytest.raises(ValueError, match="requires 'program'"):
        CompiledProgram.from_dir(tmp_path)
    assert CompiledArtifact.from_dir(tmp_path).metadata == meta


@pytest.mark.parametrize("aliases", [(True,), (-1,), (5,), (1,), (1.0,)])
def test_invalid_external_return_alias(params, aliases):
    with pytest.raises(ValueError, match="Return alias"):
        validate_return_aliases(params, aliases)


def test_names_are_shared_but_l2_capacity_is_separate(params):
    with pytest.raises(ValueError, match="unique"):
        validate_parameter_names([params[0], params[0]])
    validate_return_aliases(params, (0, 2, 2))
    entries = [replace(params[0], name=f"p{i}") for i in range(MAX_KERNEL_TENSORS + 1)]
    validate_parameter_names(entries)  # Valid metadata for a higher-level entry.
    with pytest.raises(ValueError, match="capacity"):
        validate_l2_capacity(entries)


@pytest.mark.parametrize("scalar,limit", [(False, MAX_KERNEL_TENSORS), (True, MAX_KERNEL_SCALARS)])
def test_pool_capacity(params, scalar, limit):
    entries = [replace(params[int(scalar)], name=f"p{i}") for i in range(limit)]
    validate_l2_capacity(entries)
    with pytest.raises(ValueError, match="capacity"):
        validate_l2_capacity([*entries, replace(entries[0], name="extra")])


@pytest.mark.parametrize(
    "field,value",
    [
        ("return_aliases", None),
        ("return_aliases", [2, 4, True]),
        ("return_aliases", [2]),
        ("return_aliases", "2"),
        ("lowered_params", None),
        ("lowered_params", []),
        ("supported_execution_modes", []),
        ("schema", True),
    ],
)
def test_corrupt_kernel_metadata_requires_recompile(params, tmp_path, field, value):
    record = metadata(params).record()
    record[field] = value
    (tmp_path / "compiled_meta.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="recompile"):
        CompiledArtifact.from_dir(tmp_path)


@pytest.mark.parametrize("field", ["return_aliases", "lowered_params", "supported_execution_modes"])
def test_missing_kernel_metadata_requires_recompile(params, tmp_path, field):
    record = metadata(params).record()
    del record[field]
    (tmp_path / "compiled_meta.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="recompile"):
        CompiledArtifact.from_dir(tmp_path)


def test_kernel_metadata_rejects_scalar_outputs(params, tmp_path):
    record = metadata(params).record()
    for field in ("params", "lowered_params"):
        record[field][1]["direction"] = "Out"
    (tmp_path / "compiled_meta.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="scalar.*direction In"):
        CompiledArtifact.from_dir(tmp_path)


@pytest.mark.parametrize("schema", [1, 2])
def test_legacy_program_only_metadata(params, tmp_path, schema):
    record = metadata(params, mode=ArtifactExecutionMode.PROGRAM).record()
    record["schema"] = schema
    del record["return_aliases"], record["lowered_params"]
    if schema == 1:
        del record["supported_execution_modes"]
    (tmp_path / "compiled_meta.json").write_text(json.dumps(record))
    restored = CompiledProgram.from_dir(tmp_path)
    assert restored.param_names == [p.name for p in params]
    assert restored._artifact.metadata is not None
    assert restored._artifact.metadata.return_aliases is None
    with pytest.raises(ValueError, match="requires 'kernel'"):
        restored.execution_capabilities.require(ArtifactExecutionMode.KERNEL)


@pytest.mark.parametrize(
    "dtype,shape,blocked",
    [
        (DataType.FP16, (32, 64), (1, 4, 2, 16, 16)),
        (DataType.BF16, (-1, 32, 64), (-1, 4, 2, 16, 16)),
        (DataType.INT8, (32, 64), (1, 2, 2, 16, 32)),
    ],
)
def test_nz_lowering_preserves_elements_and_bytes(dtype, shape, blocked):
    logical = ParamInfo("x", ParamDirection.In, shape, dtype, "NZ")
    lowered = replace(logical, shape=blocked)
    validate_lowered_signature([logical], [lowered])
    with pytest.raises(ValueError, match="storage"):
        validate_lowered_signature([logical], [replace(lowered, shape=(*blocked[:-1], 64))])


@pytest.mark.parametrize(
    "change",
    [dict(name="other"), dict(direction=ParamDirection.Out), dict(dtype=DataType.INT32), dict(shape=[4, 2])],
)
def test_lowering_cannot_change_external_signature(params, change):
    with pytest.raises(ValueError, match="Lowering changed"):
        validate_lowered_signature(params, [replace(params[0], **change), *params[1:]])


def test_limits_match_actual_pinned_sdk(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "abi.cpp"
    executable = tmp_path / "abi"
    source.write_text(
        '#include "task_args.h"\n#include <iostream>\n'
        f"static_assert(CHIP_MAX_TENSOR_ARGS == {MAX_KERNEL_TENSORS});\n"
        f"static_assert(CHIP_MAX_SCALAR_ARGS == {MAX_KERNEL_SCALARS});\n"
        "static_assert(MAX_TENSOR_DIMS == 5);\n"
        "static_assert(sizeof(((ChipStorageTaskArgs*)nullptr)->scalars_[0]) == 8);\n"
        "int main() { std::cout << to_u64(float(-3.5)) << ' ' << to_u64(int32_t(-7)) "
        "<< ' ' << to_u64(int8_t(-1)); }\n"
    )
    result = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-o",
            str(executable),
            "-I",
            str(root / "runtime/src/common/task_interface"),
            "-I",
            str(root / "runtime/src/common/platform/include"),
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    actual = subprocess.run([str(executable)], check=True, capture_output=True, text=True).stdout.split()
    expected = [
        int.from_bytes(struct.pack("<" + fmt, value), "little")
        for fmt, value in (("f", -3.5), ("i", -7), ("b", -1))
    ]
    assert [int(value) for value in actual] == expected


def test_metadata_import_and_restore_do_not_load_optional_runtime(run_without_optional_runtime):
    result = run_without_optional_runtime("""
import tempfile
from pathlib import Path
from pypto.ir._artifact_metadata import ArtifactMetadata
from pypto.ir.compiled_artifact import CompiledArtifact
from pypto.backend import BackendType
from pypto._artifact_contract import ArtifactExecutionMode, ExecutionCapabilities
with tempfile.TemporaryDirectory() as directory:
    metadata = ArtifactMetadata((), (), 0, 'a2a3sim', BackendType.Ascend910,
                                (), ExecutionCapabilities((ArtifactExecutionMode.KERNEL,)))
    metadata.write(Path(directory))
    artifact = CompiledArtifact.from_dir(directory)
    assert artifact.metadata == metadata
    assert artifact._chip_callable is None
""")
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
