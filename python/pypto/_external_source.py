# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Content fingerprints for hand-written external kernel sources."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.MULTILINE)

# Function attr used to carry JIT-only external-kernel include directories
# through the generated @pl.program IR. Function attrs support scalar strings,
# so the ordered path list is encoded as JSON and decoded by the backend before
# emitting kernel_config.py.
EXTERNAL_INCLUDE_DIRS_ATTR = "external_include_dirs"


def encode_external_include_dirs(include_dirs: Iterable[str | Path]) -> str:
    """Encode ordered external-kernel include directories for an IR attr."""
    return json.dumps([str(path) for path in include_dirs], separators=(",", ":"))


def decode_external_include_dirs(value: object) -> tuple[str, ...]:
    """Decode and validate external-kernel include directories from an IR attr."""
    if value is None:
        return ()
    if not isinstance(value, str):
        raise ValueError(f"{EXTERNAL_INCLUDE_DIRS_ATTR} must be a JSON string, got {type(value).__name__}")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid {EXTERNAL_INCLUDE_DIRS_ATTR} JSON: {e.msg}") from e
    if not isinstance(decoded, list) or not all(isinstance(path, str) for path in decoded):
        raise ValueError(f"{EXTERNAL_INCLUDE_DIRS_ATTR} must encode a list of strings, got {decoded!r}")
    return tuple(decoded)


def collect_external_source_files(
    sources: Iterable[str | Path],
    *,
    include_dirs: Iterable[str | Path] = (),
) -> tuple[Path, ...]:
    """Return each source and its recursively resolved local/compiler includes."""
    search_dirs = tuple(Path(path).resolve() for path in include_dirs)
    ordered: list[Path] = []
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        if not resolved.is_file():
            raise FileNotFoundError(f"External kernel source file not found: {resolved}")
        visited.add(resolved)
        ordered.append(resolved)
        text = resolved.read_text(errors="ignore")
        for delimiter, include in _INCLUDE_RE.findall(text):
            local_candidates = (resolved.parent / include,) if delimiter == '"' else ()
            candidates = (*local_candidates, *(directory / include for directory in search_dirs))
            dependency = next((candidate for candidate in candidates if candidate.is_file()), None)
            if dependency is not None:
                visit(dependency)

    for source in sources:
        visit(Path(source))
    return tuple(ordered)


def external_source_digest(
    sources: Sequence[str | Path],
    *,
    metadata: Iterable[str] = (),
    include_dirs: Iterable[str | Path] = (),
) -> str:
    """Hash external source contents, local include closure, and semantic metadata."""
    digest = hashlib.sha256()

    def update(label: str, payload: bytes) -> None:
        digest.update(label.encode())
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)

    for item in metadata:
        update("metadata", str(item).encode())
    for path in collect_external_source_files(sources, include_dirs=include_dirs):
        # Cache artifacts may move between build and production hosts. Keep the
        # logical filename, but never bake the machine-specific absolute path
        # into the digest; include directives in the source preserve the rest
        # of the logical include identity.
        update("name", path.name.encode())
        update("content", path.read_bytes())
    return digest.hexdigest()


def kernel_binary_cache_path(
    cache_dir: str | Path,
    *,
    source: str | Path,
    core_type: str,
    func_id: int | str,
    platform: str,
    external: bool,
    pto_isa_root: str | Path = "",
    runtime_name: str = "",
    include_dirs: Iterable[str | Path] = (),
) -> Path:
    """Return the content-aware cache path for one compiled kernel binary."""
    source_path = Path(source)
    suffix = f"_{platform}"
    if external:
        resolved_pto_isa_root = Path(pto_isa_root).resolve()
        resolved_include_dirs = (
            resolved_pto_isa_root / "include",
            resolved_pto_isa_root / "include" / "pto",
            *(Path(path).resolve() for path in include_dirs),
        )
        fingerprint = external_source_digest(
            [source_path],
            metadata=(
                platform,
                core_type,
                runtime_name,
            ),
            include_dirs=resolved_include_dirs,
        )
        suffix = f"_{fingerprint[:20]}"
    return Path(cache_dir) / f"incore_{func_id}_{core_type}_{source_path.stem}{suffix}.bin"


__all__ = [
    "EXTERNAL_INCLUDE_DIRS_ATTR",
    "collect_external_source_files",
    "decode_external_include_dirs",
    "encode_external_include_dirs",
    "external_source_digest",
    "kernel_binary_cache_path",
]
