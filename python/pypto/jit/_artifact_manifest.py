# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Versioned artifact records; no compiler or runtime integration."""

import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from pypto._identity import IDENTITY_SCHEMA, ToolchainIdentity, _file_digest, digest_record

ARTIFACT_SCHEMA = 1
MANIFEST_NAME = "artifact_manifest.json"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


class ArtifactState(Enum):
    """Independently published stages of one compilation key."""

    GENERATED = "generated"
    BINARY_READY = "ready"


class BuildKind(Enum):
    """Compiler adapter's program kind."""

    SINGLE_CHIP = "single_chip"
    DISTRIBUTED = "distributed"


def _require_digest(value: str) -> None:
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"Expected a full lowercase SHA-256 digest, got {value!r}")


@dataclass(frozen=True)
class ArtifactKey:
    """Complete environment and request identities supplied by a compiler adapter.

    Source and specialization digests must cover all effective compilation
    inputs. The store validates their format, not inventory completeness.
    """

    environment: ToolchainIdentity
    source_digest: str
    specialization_digest: str

    def __post_init__(self) -> None:
        if (
            not self.environment.usable
            or type(self.environment.schema) is not int
            or self.environment.schema != IDENTITY_SCHEMA
        ):
            raise ValueError("Artifact key requires a usable toolchain identity with the current schema")
        for name, value in asdict(self.environment).items():
            if name not in ("schema", "failures"):
                _require_digest(value)
        _require_digest(self.source_digest)
        _require_digest(self.specialization_digest)

    def record(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible record of every key component."""
        environment = asdict(self.environment)
        del environment["failures"]
        return {
            "schema": ARTIFACT_SCHEMA,
            "environment": environment,
            "source": self.source_digest,
            "specialization": self.specialization_digest,
        }

    @property
    def digest(self) -> str:
        """Full deterministic digest, including the artifact protocol schema."""
        return digest_record(self.record())


def _relative_path(value: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid artifact relative path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in (".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"Artifact path must be normalized and relative: {value!r}")
    if value == MANIFEST_NAME:
        raise ValueError(f"Artifact payload uses reserved manifest path: {value!r}")
    return value


@dataclass(frozen=True)
class ArtifactSpec:
    """Stage and required payload paths established by the compiler adapter.

    BINARY_READY is only meaningful when the adapter lists every binary and
    loader metadata file required by its runtime. This layer cannot infer that.
    """

    state: ArtifactState
    build_kind: BuildKind
    required_files: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.state, ArtifactState) or not isinstance(self.build_kind, BuildKind):
            raise ValueError("Artifact spec requires ArtifactState and BuildKind enum values")
        required = tuple(sorted(_relative_path(path) for path in self.required_files))
        if not required or len(set(required)) != len(required):
            raise ValueError(f"Artifact required files must be nonempty and unique, got {required!r}")
        object.__setattr__(self, "required_files", required)


def check_directory(path: Path) -> None:
    """Reject non-directory ancestors, including symbolic links."""
    for ancestor in (*reversed(path.parents), path):
        if not stat.S_ISDIR(ancestor.lstat().st_mode):
            raise ValueError(f"Artifact directory or ancestor is not a real directory: {ancestor}")


def inventory(directory: Path) -> list[dict[str, Any]]:
    """Hash every regular payload file, rejecting links and special files."""
    check_directory(directory)
    files: list[dict[str, Any]] = []

    def visit(parent: Path) -> None:
        with os.scandir(parent) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
        for entry in children:
            path = Path(entry.path)
            relative = path.relative_to(directory).as_posix()
            mode = entry.stat(follow_symlinks=False).st_mode
            if relative == MANIFEST_NAME:
                if not stat.S_ISREG(mode):
                    raise ValueError(f"Artifact manifest is not a regular file: {path}")
            elif stat.S_ISDIR(mode):
                _relative_path(relative)
                visit(path)
            elif stat.S_ISREG(mode):
                size, digest = _file_digest(path)
                files.append(
                    {
                        "path": _relative_path(relative),
                        "size": size,
                        "sha256": digest,
                        "execute_bits": mode & 0o111,
                    }
                )
            else:
                raise ValueError(f"Artifact payload is not a regular file or directory: {path}")

    visit(directory)
    return sorted(files, key=lambda entry: entry["path"])


def make_manifest(directory: Path, key: ArtifactKey, spec: ArtifactSpec) -> dict[str, Any]:
    """Inventory a completed private build and verify its required files."""
    files = inventory(directory)
    missing = set(spec.required_files) - {entry["path"] for entry in files}
    if missing:
        raise ValueError(f"Artifact build is missing required files: {sorted(missing)}")
    return {
        "schema": ARTIFACT_SCHEMA,
        "key": key.digest,
        "components": key.record(),
        "state": spec.state.value,
        "build_kind": spec.build_kind.value,
        "required_files": list(spec.required_files),
        "files": files,
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate artifact manifest field: {name!r}")
        result[name] = value
    return result


def read_manifest(directory: Path, key: ArtifactKey, spec: ArtifactSpec) -> dict[str, Any]:
    """Validate a completion marker against the exact payload and request.

    Comparison uses canonical JSON so booleans/floats cannot impersonate
    integer protocol fields. Paths from the marker are never opened.
    """
    check_directory(directory)
    path = directory / MANIFEST_NAME
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Artifact manifest must be a regular file: {path}")
    with path.open("rb") as stream:
        raw = stream.read(_MAX_MANIFEST_BYTES + 1)
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ValueError(f"Artifact manifest exceeds {_MAX_MANIFEST_BYTES} bytes: {path}")
    actual = json.loads(raw, object_pairs_hook=_unique_object)
    expected = make_manifest(directory, key, spec)
    if encode_manifest(actual) != encode_manifest(expected):
        raise ValueError(f"Artifact manifest does not match the request or payload: {path}")
    return expected


def encode_manifest(record: Any) -> bytes:
    """Serialize a manifest strictly and deterministically."""
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise ValueError(f"Artifact manifest exceeds {_MAX_MANIFEST_BYTES} bytes")
    return encoded
