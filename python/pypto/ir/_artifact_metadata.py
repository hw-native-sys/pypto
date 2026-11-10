# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared sidecar parsing and single-chip artifact metadata; no native loading."""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from pypto._artifact_contract import ArtifactExecutionMode, ExecutionCapabilities
from pypto.backend import BackendType

from .param_info import (
    _is_int_list,
    _param_info_from_dict,
    _param_info_to_dict,
    _ParamInfo,
    validate_parameter_names,
    validate_return_aliases,
)

# Filename of the small JSON sidecar persisted alongside the build artifacts so
# a single-orchestration program can be reconstructed (``from_dir``) without the
# live IR -- the L2 counterpart of ``distributed_meta.json``. Bump
# ``_COMPILED_META_SCHEMA`` on any incompatible format change.
#
# Both sidecars embed the same ``_param_info_to_dict`` payload but version it
# independently, so a change to *that* shared format must bump this schema AND
# ``distributed_compiled_program._META_SCHEMA``.
_COMPILED_META_FILENAME = "compiled_meta.json"
_COMPILED_META_SCHEMA = 3

# The L3 counterpart, kept here so both names sit next to the marker table below;
# ``distributed_compiled_program`` re-exports it and owns its schema constant.
_DISTRIBUTED_META_FILENAME = "distributed_meta.json"

# What a consumer reads to decide which *kind* of build a directory holds:
#
#   L2 single-chip -- top-level ``kernel_config.py``   (+ ``compiled_meta.json``)
#   L3 distributed -- ``orchestration/host_orch.py``   (+ ``distributed_meta.json``)
#
# ``replay()`` takes the L3 path exactly when there is no top-level
# ``kernel_config.py`` *and* ``host_orch.py`` is present (see
# ``runtime/debug/replay.py``). ``ir.compile`` never clears ``output_dir``, so a
# marker an earlier compile of the *other* kind left behind does not merely go
# stale -- it re-points the whole directory at that older build: an L2 leftover
# makes a fresh L3 build replay as L2, and an L3 leftover keeps
# ``DistributedCompiledProgram.from_dir`` loadable on top of L2 artifacts. Every
# fresh compile therefore drops the markers its own kind does not write (see
# :func:`_drop_foreign_build_markers`).
_BUILD_KIND_MARKERS = (
    _COMPILED_META_FILENAME,
    _DISTRIBUTED_META_FILENAME,
    "kernel_config.py",
    "orchestration/host_orch.py",
    "orchestration/cluster_orch.py",
)


def _is_obj_list(v: object) -> TypeGuard[list[object]]:
    return isinstance(v, list)


def _is_str_dict(v: object) -> TypeGuard[dict[str, object]]:
    """A JSON object: ``json.loads`` only produces string keys."""
    return isinstance(v, dict)


def _remove_meta(output_dir: Path, filename: str = _COMPILED_META_FILENAME) -> None:
    """Drop ``<output_dir>/<filename>`` if present.

    ``ir.compile`` never clears ``output_dir`` (``os.makedirs(exist_ok=True)``),
    so recompiling a *different* program shape into a reused directory can leave
    a sidecar describing the previous one. Every fresh-compile path that does
    not overwrite the sidecar must therefore remove it: a stale signature would
    drive the new artifacts with the old parameter ABI, which
    :meth:`CompiledProgram.from_dir` has no way to detect.

    Shared with the L3 ``distributed_meta.json``, which carries the same
    contract; *filename* selects which sidecar to drop.
    """
    (output_dir / filename).unlink(missing_ok=True)


def _drop_foreign_build_markers(output_dir: Path, keep: Sequence[str]) -> None:
    """Delete every build-kind marker in *output_dir* that this compile does not write.

    *keep* names the markers the current build owns (see
    :data:`_BUILD_KIND_MARKERS`); everything else is necessarily a leftover of a
    compile of a different kind into the same reused directory, and leaving it
    behind silently redirects consumers to that older build rather than merely
    aging out:

    - an L2 build's top-level ``kernel_config.py`` makes ``replay()`` treat a
      later L3 build as single-chip and run the stale L2 artifacts;
    - an L3 build's ``distributed_meta.json`` / ``host_orch.py`` keep
      ``DistributedCompiledProgram.from_dir`` (and the L3 replay path) loadable
      on a directory whose top level is now an L2 build.

    Only markers are removed, never artifact trees: whatever a consumer reaches
    for afterwards is either this build's or absent, so a stale directory fails
    loudly instead of running the wrong program.
    """
    for marker in _BUILD_KIND_MARKERS:
        if marker not in keep:
            (output_dir / marker).unlink(missing_ok=True)


def _write_meta_atomically(target: Path, meta: dict[str, Any]) -> None:
    """Serialise *meta* to *target* via a temp file and :func:`os.replace`.

    Atomic so a reader never observes a half-written sidecar, and so a crash
    mid-write leaves the previous file intact rather than a truncated one — a
    truncated sidecar is exactly the input the loaders must otherwise reject.
    The temp name carries the writing pid, so concurrent compiles into one
    directory cannot clobber each other's intermediate file.

    Shared by the L2 ``compiled_meta.json`` and L3 ``distributed_meta.json``
    writers.
    """
    tmp = target.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, target)


def _meta_error(filename: str, meta_path: Path, detail: str) -> ValueError:
    """Build the shared "this sidecar is not loadable" error.

    One message shape for both sidecars: name the file, say what is wrong, and
    point at the recompile that regenerates it.
    """
    return ValueError(
        f"Invalid {filename} in {meta_path}: {detail}. The metadata was "
        f"written by a different pypto version or hand-edited — recompile via "
        f"ir.compile() to refresh."
    )


def _read_capabilities(
    meta: dict[str, Any], found_schema: int, filename: str, require_program: bool, bad: Any
) -> ExecutionCapabilities:
    try:
        capabilities = (
            ExecutionCapabilities()
            if found_schema == 1
            and filename == _COMPILED_META_FILENAME
            and "supported_execution_modes" not in meta
            else ExecutionCapabilities.from_record(meta.get("supported_execution_modes"))
        )
        if require_program:
            capabilities.require(ArtifactExecutionMode.PROGRAM)
        if filename == _COMPILED_META_FILENAME and found_schema < 3:
            if capabilities != ExecutionCapabilities():
                raise ValueError("Legacy metadata cannot establish kernel capability; recompile")
    except ValueError as exc:
        raise bad(str(exc)) from exc

    return capabilities


def _load_meta(
    meta_path: Path, *, filename: str, schema: int | tuple[int, ...], require_program: bool = True
) -> dict[str, Any]:
    """Read and fully validate the fields the L2 and L3 sidecars share.

    Every malformed-payload failure surfaces as a single :class:`ValueError`
    naming *meta_path* and the recompile fix, rather than leaking whichever
    ``JSONDecodeError`` / ``KeyError`` / ``AttributeError`` / ``TypeError`` the
    offending field happened to raise. A build directory is user-supplied
    input, so a hand-edited or truncated sidecar must fail like bad input, not
    like an internal error.

    Args:
        meta_path: The sidecar to read.
        filename: Its bare filename, for error messages.
        schema: The schema version this build understands. The two sidecars
            version independently, so each caller passes its own.

    Returns:
        ``{"param_infos", "num_return_types", "platform", "backend_type"}``
        already converted to their runtime types, plus ``"raw"`` — the parsed
        JSON object, so a caller can validate the fields only its own sidecar
        carries (e.g. the L3 ``distributed_config``).

    Raises:
        ValueError: the file is not readable as the expected schema.
    """

    def _bad(detail: str) -> ValueError:
        return _meta_error(filename, meta_path, detail)

    try:
        raw_text = meta_path.read_text()
    except OSError as exc:  # unreadable / vanished / replaced by a directory
        raise _bad(f"cannot be read ({exc})") from exc
    except UnicodeDecodeError as exc:  # binary or mis-encoded file
        raise _bad(f"is not valid UTF-8 text ({exc})") from exc
    try:
        raw_meta: object = json.loads(raw_text)
    except ValueError as exc:  # JSONDecodeError is a ValueError subclass
        raise _bad(f"not valid JSON ({exc})") from exc
    if not _is_str_dict(raw_meta):
        raise _bad(f"expected a JSON object, got {type(raw_meta).__name__}")
    meta = raw_meta

    found_schema = meta.get("schema")
    if type(found_schema) is not int or found_schema not in (
        (1, 2, 3)
        if filename == _COMPILED_META_FILENAME
        else schema
        if isinstance(schema, tuple)
        else (schema,)
    ):
        raise ValueError(
            f"Incompatible {filename} schema {found_schema!r} (expected "
            f"{schema}) in {meta_path}. The metadata was written by a "
            f"different pypto version — recompile via ir.compile() to refresh."
        )

    capabilities = _read_capabilities(meta, found_schema, filename, require_program, _bad)

    raw_params: object = meta.get("params")
    if not _is_obj_list(raw_params):
        raise _bad(f"'params' must be a list, got {type(raw_params).__name__}")
    param_infos: list[_ParamInfo] = []
    for i, entry in enumerate(raw_params):
        if not _is_str_dict(entry):
            raise _bad(f"params[{i}] must be an object, got {type(entry).__name__}")
        try:
            param_infos.append(_param_info_from_dict(entry))
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise _bad(f"params[{i}] is malformed ({exc})") from exc

    try:
        validate_parameter_names(param_infos)
    except ValueError as exc:
        raise _bad(str(exc)) from exc

    num_return_types = meta.get("num_return_types", 0)
    if not isinstance(num_return_types, int) or isinstance(num_return_types, bool) or num_return_types < 0:
        raise _bad(f"'num_return_types' must be a non-negative integer, got {num_return_types!r}")

    # Strict enum-member lookup, not getattr(): ``getattr(BackendType, "mro")``
    # resolves to a bound method and would sail through as a bogus backend.
    raw_backend = meta.get("backend_type", "Ascend910")
    backend_type = BackendType.__members__.get(raw_backend) if isinstance(raw_backend, str) else None
    if backend_type is None:
        raise _bad(f"'backend_type' must be one of {sorted(BackendType.__members__)}, got {raw_backend!r}")

    platform = meta.get("platform")
    if platform is not None and not isinstance(platform, str):
        raise _bad(f"'platform' must be a string or absent, got {type(platform).__name__}")

    return {
        "execution_capabilities": capabilities,
        "param_infos": param_infos,
        "num_return_types": num_return_types,
        "platform": platform,
        "backend_type": backend_type,
        "raw": meta,
    }


@dataclass(frozen=True)
class ArtifactMetadata:
    """Logical caller signature and lowered assembly signature of one entry.

    None aliases mean unavailable analysis (legacy/program-only), never an
    inferred kernel contract. Alias indices address the full parameter list.
    """

    parameters: tuple[_ParamInfo, ...]
    lowered_parameters: tuple[_ParamInfo, ...]
    num_return_types: int
    platform: str
    backend_type: BackendType
    return_aliases: tuple[int, ...] | None = None
    execution_capabilities: ExecutionCapabilities = ExecutionCapabilities()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "lowered_parameters", tuple(self.lowered_parameters))
        if type(self.num_return_types) is not int or self.num_return_types < 0:
            raise ValueError("num_return_types must be a non-negative integer")
        validate_parameter_names(self.parameters)
        validate_parameter_names(self.lowered_parameters)
        if self.return_aliases is not None:
            object.__setattr__(self, "return_aliases", tuple(self.return_aliases))
            validate_return_aliases(self.parameters, self.return_aliases)
            if len(self.return_aliases) != self.num_return_types:
                raise ValueError("Return alias count does not match num_return_types; recompile")
        if ArtifactExecutionMode.KERNEL in self.execution_capabilities.modes:
            if self.return_aliases is None:
                raise ValueError("Kernel metadata requires return aliases; recompile")
            from ._compile_contract import (  # noqa: PLC0415
                validate_kernel_parameters,
                validate_lowered_signature,
            )

            validate_kernel_parameters(self.parameters)
            validate_lowered_signature(self.parameters, self.lowered_parameters)

    def record(self) -> dict[str, Any]:
        return {
            "schema": _COMPILED_META_SCHEMA,
            "supported_execution_modes": self.execution_capabilities.record(),
            "params": [_param_info_to_dict(p) for p in self.parameters],
            "lowered_params": [_param_info_to_dict(p) for p in self.lowered_parameters],
            "num_return_types": self.num_return_types,
            "return_aliases": None if self.return_aliases is None else list(self.return_aliases),
            "platform": self.platform,
            "backend_type": self.backend_type.name,
        }

    def write(self, directory: Path) -> None:
        _write_meta_atomically(directory / _COMPILED_META_FILENAME, self.record())

    @classmethod
    def read(cls, directory: Path) -> "ArtifactMetadata":
        path = directory / _COMPILED_META_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; recompile via ir.compile() to refresh")
        meta = _load_meta(
            path, filename=_COMPILED_META_FILENAME, schema=_COMPILED_META_SCHEMA, require_program=False
        )
        raw = meta["raw"]
        try:
            if ArtifactExecutionMode.KERNEL in meta["execution_capabilities"].modes:
                required = {
                    "params",
                    "lowered_params",
                    "return_aliases",
                    "num_return_types",
                    "platform",
                    "backend_type",
                }
                if required - raw.keys() or raw.get("platform") is None:
                    raise ValueError("Kernel metadata is incomplete; recompile")
            if raw["schema"] == _COMPILED_META_SCHEMA:
                raw_aliases: object = raw["return_aliases"]
                if raw_aliases is not None and not _is_int_list(raw_aliases):
                    raise ValueError("return_aliases must be a list of integer indices or null")
                aliases = None if raw_aliases is None else tuple(raw_aliases)
                raw_lowered: object = raw["lowered_params"]
                if not _is_obj_list(raw_lowered):
                    raise ValueError("lowered_params must be a list")
                lowered_params: list[_ParamInfo] = []
                for item in raw_lowered:
                    if not _is_str_dict(item):
                        raise ValueError("lowered_params entries must be objects")
                    lowered_params.append(_param_info_from_dict(item))
                lowered = tuple(lowered_params)
            else:
                aliases, lowered = None, tuple(meta["param_infos"])
            platform = meta["platform"]
            if platform is None:
                from pypto.pypto_core import backend  # noqa: PLC0415

                platform = (
                    backend.get_backend_instance(meta["backend_type"])
                    .get_handler()
                    .get_default_sim_platform()
                )
            return cls(
                tuple(meta["param_infos"]),
                lowered,
                meta["num_return_types"],
                platform,
                meta["backend_type"],
                aliases,
                meta["execution_capabilities"],
            )
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise _meta_error(_COMPILED_META_FILENAME, path, str(exc)) from exc


# Shared by single-chip and distributed compilation.
__all__ = ["_remove_meta", "_drop_foreign_build_markers"]
