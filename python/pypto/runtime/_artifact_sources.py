# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit packaging of generated configurations and supported extern sources."""

import os
import re
from pathlib import Path
from types import ModuleType
from typing import Any

from pypto.jit._artifact_manifest import BuildKind, encode_manifest

from ._prebuilt import _encode_signature, _relative_file, chip_directories

_INCLUDE = re.compile(r"^\s*#\s*include\s*(.*)$", re.MULTILINE)
_LITERAL_INCLUDE = re.compile(r'([<"])([^>"]+)[>"](?:\s*(?://.*|/\*.*\*/))?\s*$')


def read_kernel_config(path: Path) -> ModuleType:
    """Execute trusted generated configuration without reading or writing bytecode."""
    module = ModuleType("_pypto_artifact_config")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def _external_files(source: Path, include_dirs: tuple[Path, ...]) -> dict[Path, bytes]:
    files: dict[Path, bytes] = {}

    def visit(path: Path) -> None:
        path = path.resolve(strict=True)
        if path in files:
            return
        data = path.read_bytes()
        files[path] = data
        for include in _INCLUDE.findall(data.decode("utf-8")):
            match = _LITERAL_INCLUDE.fullmatch(include)
            if match is None:
                raise ValueError(f"Artifact extern includes must be literal: {path}: {include}")
            delimiter, name = match.groups()
            if Path(name).is_absolute() or "\\" in name:
                raise ValueError(f"Artifact extern includes must be relative: {path}: {name}")
            candidates = (path.parent / name,) if delimiter == '"' else ()
            candidates += tuple(root / name for root in include_dirs)
            dependency = next((p for p in candidates if p.is_file()), None)
            if dependency is not None:
                visit(dependency)
            elif delimiter == '"':
                raise ValueError(f"Unresolved local extern include: {path}: {name}")
            # Unresolved angle includes belong to the separately identified SDK.

    visit(source)
    return files


def _package_external(root: Path, kernel: dict[str, Any], index: int) -> None:
    source = Path(kernel["source"]).resolve(strict=True)
    includes = tuple(Path(p).resolve(strict=True) for p in kernel.get("extra_include_dirs", ()))
    files = _external_files(source, includes)
    common = Path(os.path.commonpath([str(p.parent) for p in files] + [str(p) for p in includes]))
    destination = root / "extern" / str(index)
    for path, data in files.items():
        target = destination / path.relative_to(common)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    relocated = destination / source.relative_to(common)
    kernel["source"] = str(relocated.relative_to(root))
    relocated_includes = [destination / path.relative_to(common) for path in includes]
    for path in relocated_includes:
        path.mkdir(parents=True, exist_ok=True)
    kernel["extra_include_dirs"] = [str(path.relative_to(root)) for path in relocated_includes]


def _local_source(root: Path, source: str) -> str:
    path = Path(source).resolve(strict=True)
    if root not in path.parents:
        raise ValueError(f"Generated source is outside the artifact: {source}")
    relative = str(path.relative_to(root))
    _relative_file(root, relative)
    return relative


def package_generated_sources(directory: Path, kind: BuildKind) -> None:
    """Make a private generated tree self-contained before GENERATED publication.

    Literal local extern include graphs are supported. Macro/absolute includes
    and missing quoted dependencies fail closed. SDK angle includes remain
    supplied by the explicitly identified toolchain. This mutates only the
    caller's private tree, never an ArtifactHandle's published directory.
    """
    directory = directory.resolve()
    if (directory / "artifact_manifest.json").exists():
        raise ValueError("Cannot package sources inside a published artifact")
    for chip in chip_directories(directory, kind).values():
        config = read_kernel_config(chip / "kernel_config.py")
        kernels = [dict(k) for k in config.KERNELS]
        for index, kernel in enumerate(kernels):
            if kernel.get("external", False):
                _package_external(chip, kernel, index)
            else:
                kernel["source"] = _local_source(chip, kernel["source"])
            kernel["signature"] = _encode_signature(kernel.get("signature", []))
        orchestration = dict(config.ORCHESTRATION)
        orchestration["source"] = _local_source(chip, orchestration["source"])
        orchestration["signature"] = _encode_signature(orchestration.get("signature", []))
        runtime_config = getattr(config, "RUNTIME_CONFIG", {})
        # Reject non-serializable build inputs instead of persisting repr(object).
        encode_manifest({"kernels": kernels, "orchestration": orchestration, "runtime": runtime_config})
        text = (
            "from pathlib import Path\n"
            "from simpler.task_interface import ArgDirection as _D\n"
            "_ROOT_DIR = Path(__file__).parent\n"
            f"KERNELS = {kernels!r}\n"
            f"ORCHESTRATION = {orchestration!r}\n"
            f"RUNTIME_CONFIG = {runtime_config!r}\n"
            "for _entry in [*KERNELS, ORCHESTRATION]:\n"
            "    _entry['source'] = str(_ROOT_DIR / _entry['source'])\n"
            "    _entry['signature'] = [getattr(_D, s) for s in _entry['signature']]\n"
            "    if 'extra_include_dirs' in _entry:\n"
            "        _entry['extra_include_dirs'] = [\n"
            "            str(_ROOT_DIR / p) for p in _entry['extra_include_dirs']]\n"
        )
        (chip / "kernel_config.py").write_text(text, encoding="utf-8")


def validate_generated_sources(directory: Path, kind: BuildKind) -> None:
    """Reject generated handles that still depend on an external source tree."""
    for chip in chip_directories(directory, kind).values():
        config = read_kernel_config(chip / "kernel_config.py")
        for entry in [*config.KERNELS, config.ORCHESTRATION]:
            _local_source(chip, entry["source"])
            for include in entry.get("extra_include_dirs", ()):
                path = Path(include).resolve(strict=True)
                if path != chip and chip not in path.parents:
                    raise ValueError(f"External include directory is outside the artifact: {path}")
