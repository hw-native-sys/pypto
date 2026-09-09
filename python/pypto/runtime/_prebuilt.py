# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal device-binary records and read-only callable reconstruction."""

import hashlib
import json
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from pypto.jit._artifact_manifest import (
    ArtifactSpec,
    ArtifactState,
    BuildKind,
    _unique_object,
    encode_manifest,
)

BINARY_MANIFEST = "binary_manifest.json"
_BINARY_SCHEMA = 1
_DIRECTIONS = {"SCALAR", "IN", "OUT", "INOUT"}


def _relative_file(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid prebuilt relative file: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(p in (".", "..") for p in path.parts):
        raise ValueError(f"Invalid prebuilt relative file: {value!r}")
    target = root / value
    parents = [root]
    for part in path.parts[:-1]:
        parents.append(parents[-1] / part)
    for parent in parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ValueError(f"Prebuilt ancestor must be a real directory: {parent}")
    if not stat.S_ISREG(target.lstat().st_mode):
        raise ValueError(f"Prebuilt payload must be a regular file: {target}")
    return target


def _signature(value: Any) -> list[str]:
    if not isinstance(value, list) or any(type(v) is not str or v not in _DIRECTIONS for v in value):
        raise ValueError(f"Invalid prebuilt signature: {value!r}")
    return value


def _encode_signature(value: Any) -> list[str]:
    return _signature([v if isinstance(v, str) else v.name for v in value])


def _save_bytes(root: Path, name: str, data: bytes) -> dict[str, Any]:
    if not data:
        raise ValueError(f"Empty prebuilt binary: {name}")
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_chip_binaries(
    directory: Path,
    platform: str,
    orchestration: dict[str, Any],
    kernels: list[tuple[dict[str, Any], bytes]],
    orchestration_binary: bytes,
    runtime_name: str,
    runtime_config: dict[str, Any],
) -> None:
    """Record final device bytes in a private build, after successful compilation."""
    record = {
        "schema": _BINARY_SCHEMA,
        "platform": platform,
        "runtime_name": runtime_name,
        "runtime_config": runtime_config,
        "orchestration": {
            "function_name": orchestration["function_name"],
            "display_name": Path(orchestration["source"]).stem,
            "signature": _encode_signature(orchestration.get("signature", [])),
            "binary": _save_bytes(directory, "prebuilt/orchestration.bin", orchestration_binary),
        },
        "kernels": [
            {
                "func_id": kernel["func_id"],
                "name": kernel.get("name", str(kernel["func_id"])),
                "signature": _encode_signature(kernel.get("signature", [])),
                "binary": _save_bytes(directory, f"prebuilt/kernel_{index}.bin", binary),
            }
            for index, (kernel, binary) in enumerate(kernels)
        ],
    }
    (directory / BINARY_MANIFEST).write_bytes(encode_manifest(record))


def chip_directories(directory: Path, kind: BuildKind) -> dict[str, Path]:
    """Enumerate all supported chip sub-builds without executing configuration."""
    if kind is BuildKind.SINGLE_CHIP:
        if not (directory / "kernel_config.py").is_file():
            raise ValueError(f"Single-chip artifact lacks kernel_config.py: {directory}")
        return {".": directory}
    children = directory / "next_levels"
    result = {}
    for child in sorted(children.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "kernel_config.py").is_file():
            raise ValueError(f"Distributed chip lacks kernel_config.py: {child}")
        result[child.name] = child
    if not result:
        raise ValueError(f"Distributed artifact has no chip builds: {children}")
    return result


def prepare_prebuilt(directory: Path, platform: str, kind: BuildKind) -> None:
    """Finish every chip binary without constructing a device or executing kernels."""
    from .device_runner import _compile_and_assemble  # noqa: PLC0415

    chips = chip_directories(directory, kind)
    for chip in chips.values():
        _compile_and_assemble(chip, platform, save_prebuilt=True)
    if kind is BuildKind.DISTRIBUTED:
        record = {"schema": _BINARY_SCHEMA, "platform": platform, "chips": list(chips)}
        (directory / BINARY_MANIFEST).write_bytes(encode_manifest(record))
    # Verify completeness before the caller is allowed to publish READY.
    read_prebuilt(directory, platform, kind)


def ready_spec(directory: Path, generated: ArtifactSpec) -> ArtifactSpec:
    """Declare every final binary and child manifest before looking up READY."""
    from ._artifact_sources import read_kernel_config  # noqa: PLC0415

    required = set(generated.required_files) | {BINARY_MANIFEST}
    for chip in chip_directories(directory, generated.build_kind).values():
        prefix = chip.relative_to(directory)
        config = read_kernel_config(chip / "kernel_config.py")
        required.update(str(prefix / path) for path in (BINARY_MANIFEST, "prebuilt/orchestration.bin"))
        required.update(str(prefix / "prebuilt" / f"kernel_{i}.bin") for i in range(len(config.KERNELS)))
    return ArtifactSpec(ArtifactState.BINARY_READY, generated.build_kind, tuple(sorted(required)))


def _read_json(root: Path) -> dict[str, Any]:
    with _relative_file(root, BINARY_MANIFEST).open("rb") as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError(f"Prebuilt manifest is too large: {root}")
    record = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(record, dict) or type(record.get("schema")) is not int or record["schema"] != 1:
        raise ValueError(f"Unsupported prebuilt manifest schema: {root}")
    return record


def _binary(root: Path, record: Any) -> bytes:
    if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
        raise ValueError(f"Invalid prebuilt binary record in {root}")
    if type(record["size"]) is not int or record["size"] <= 0:
        raise ValueError(f"Invalid prebuilt binary size: {record['size']!r}")
    path = _relative_file(root, record["path"])
    if path.stat().st_size != record["size"]:
        raise ValueError(f"Prebuilt binary size mismatch: {path}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"Prebuilt binary digest mismatch: {path}")
    return data


def _read_chip(root: Path, platform: str) -> dict[str, Any]:
    record = _read_json(root)
    if set(record) != {"schema", "platform", "runtime_name", "runtime_config", "orchestration", "kernels"}:
        raise ValueError(f"Invalid prebuilt chip fields: {root}")
    if record["platform"] != platform:
        raise ValueError(f"Prebuilt platform {record['platform']!r} does not match {platform!r}")
    if not isinstance(record["runtime_name"], str) or not record["runtime_name"]:
        raise ValueError(f"Invalid prebuilt runtime name: {root}")
    config = record["runtime_config"]
    if (
        not isinstance(config, dict)
        or config.get("runtime", record["runtime_name"]) != record["runtime_name"]
    ):
        raise ValueError(f"Invalid prebuilt runtime configuration: {root}")
    orch = record["orchestration"]
    if not isinstance(orch, dict) or set(orch) != {"function_name", "display_name", "signature", "binary"}:
        raise ValueError(f"Invalid prebuilt orchestration: {root}")
    for name in ("function_name", "display_name"):
        if not isinstance(orch[name], str) or not orch[name] or "\x00" in orch[name]:
            raise ValueError(f"Invalid prebuilt orchestration {name}: {root}")
    _signature(orch["signature"])
    orch["binary"] = _binary(root, orch["binary"])
    if not isinstance(record["kernels"], list):
        raise ValueError(f"Invalid prebuilt kernel list: {root}")
    seen: set[int] = set()
    for kernel in record["kernels"]:
        if not isinstance(kernel, dict) or set(kernel) != {"func_id", "name", "signature", "binary"}:
            raise ValueError(f"Invalid prebuilt kernel record: {root}")
        fid = kernel["func_id"]
        if type(fid) is not int or not 0 <= fid < 2**31 or fid in seen:
            raise ValueError(f"Invalid or duplicate prebuilt function ID: {fid!r}")
        seen.add(fid)
        if not isinstance(kernel["name"], str) or not kernel["name"]:
            raise ValueError(f"Invalid prebuilt kernel name: {root}")
        _signature(kernel["signature"])
        kernel["binary"] = _binary(root, kernel["binary"])
    return record


def read_prebuilt(directory: Path, platform: str, kind: BuildKind) -> dict[str, dict[str, Any]]:
    """Read and validate every child before any callable can be assembled."""
    chips = chip_directories(directory, kind)
    if kind is BuildKind.DISTRIBUTED:
        parent = _read_json(directory)
        if (
            set(parent) != {"schema", "platform", "chips"}
            or parent["platform"] != platform
            or parent["chips"] != list(chips)
        ):
            raise ValueError(f"Prebuilt distributed manifest does not match all chip builds: {directory}")
    result = {name: _read_chip(path, platform) for name, path in chips.items()}
    if len({record["runtime_name"] for record in result.values()}) != 1:
        raise ValueError(f"Prebuilt chip builds use inconsistent runtimes: {directory}")
    return result


def load_prebuilt(
    directory: Path, platform: str, kind: BuildKind
) -> dict[str, tuple[Any, str, dict[str, Any]]]:
    """Assemble validated bytes without compiler resolution, locks, or filesystem writes.

    The artifact adapter must first validate the enclosing ArtifactHandle and
    expected key. This internal helper also validates private fallback builds.
    """
    records = read_prebuilt(directory, platform, kind)
    # Simpler's optional native interface has no static stubs (as in task_interface.py).
    from simpler.task_interface import ArgDirection  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    from ._callable_identity import register_callable_identity  # noqa: PLC0415
    from .task_interface import (  # noqa: PLC0415
        ChipCallable,  # pyright: ignore[reportAttributeAccessIssue]
        CoreCallable,  # pyright: ignore[reportAttributeAccessIssue]
    )

    result = {}
    for name, record in records.items():
        kernels = [
            (
                kernel["func_id"],
                CoreCallable.build(
                    signature=[getattr(ArgDirection, v) for v in kernel["signature"]], binary=kernel["binary"]
                ),
            )
            for kernel in record["kernels"]
        ]
        orch = record["orchestration"]
        callable_ = ChipCallable.build(
            signature=[getattr(ArgDirection, v) for v in orch["signature"]],
            func_name=orch["function_name"],
            binary=orch["binary"],
            children=kernels,
        )
        register_callable_identity(orch["binary"], orch["display_name"])
        result[name] = (callable_, record["runtime_name"], record["runtime_config"])
    return result


def kernel_name_map(directory: Path) -> dict[str, str]:
    """Read diagnostic labels from an already validated binary manifest."""
    return {str(k["func_id"]): k["name"] for k in _read_json(directory)["kernels"]}
