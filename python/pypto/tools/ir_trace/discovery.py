# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Discover ordered IR snapshots in a pass dump directory."""

import re
from pathlib import Path

from .model import IRTraceError, Snapshot, split_source_lines

_PASS_RE = re.compile(r"^(?P<index>\d+)_after_(?P<name>.+)\.py$")
_NUMERIC_PY_RE = re.compile(r"^\d+_.*\.py$")


def _input_io_error(action: str, path: Path, error: OSError) -> IRTraceError:
    reason = error.strerror or "I/O error"
    return IRTraceError(f"failed to {action} {path.name}: {reason}")


def _read_utf8(path: Path) -> str:
    try:
        if not path.is_file():
            raise IRTraceError(f"{path.name} is not a file")
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise IRTraceError(f"{path.name} is not valid UTF-8") from exc
    except OSError as exc:
        raise _input_io_error("read", path, exc) from exc


def _read_optional_utf8(path: Path) -> str | None:
    try:
        exists = path.exists()
    except OSError as exc:
        raise _input_io_error("read", path, exc) from exc
    return _read_utf8(path) if exists else None


def discover_snapshots(directory: Path) -> tuple[Snapshot, ...]:
    """Read the frontend and each consecutively numbered pass snapshot.

    Args:
        directory: The ``passes_dump/`` directory emitted by the pass manager.

    Returns:
        The frontend snapshot followed by pass snapshots in index order.

    Raises:
        IRTraceError: The directory or its snapshots are missing or malformed.
    """
    if not directory.exists():
        raise IRTraceError(f"input directory does not exist: {directory}")
    if not directory.is_dir():
        raise IRTraceError(f"input path is not a directory: {directory}")

    frontend = directory / "00_frontend.py"
    if not frontend.is_file():
        raise IRTraceError(f"missing 00_frontend.py in {directory}")

    try:
        paths = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise _input_io_error("enumerate", directory, exc) from exc

    indexed: dict[int, tuple[str, Path]] = {}
    for path in paths:
        match = _PASS_RE.fullmatch(path.name)
        if match:
            index = int(match.group("index"))
            if index == 0:
                raise IRTraceError(f"pass snapshot index must be at least 01: {path.name}")
            if index in indexed:
                previous = indexed[index][1].name
                raise IRTraceError(f"duplicate snapshot index {index:02d}: {previous} and {path.name}")
            indexed[index] = (match.group("name"), path)
        elif _NUMERIC_PY_RE.fullmatch(path.name) and path.name != "00_frontend.py":
            raise IRTraceError(f"malformed snapshot name {path.name}; expected NN_after_PassName.py")

    if not indexed:
        raise IRTraceError(f"no pass snapshots found in {directory}")

    for index in range(1, max(indexed) + 1):
        if index not in indexed:
            raise IRTraceError(f"missing snapshot index {index:02d} in {directory}")

    frontend_text = _read_utf8(frontend)
    snapshots = [
        Snapshot(
            index=0,
            pass_name=None,
            path=frontend,
            text=frontend_text,
            lines=split_source_lines(frontend_text),
        )
    ]
    for index, (name, path) in sorted(indexed.items()):
        text = _read_utf8(path)
        warning_path = path.with_suffix(".log")
        warning_text = _read_optional_utf8(warning_path)
        snapshots.append(
            Snapshot(
                index=index,
                pass_name=name,
                path=path,
                text=text,
                lines=split_source_lines(text),
                warning_text=warning_text,
            )
        )
    return tuple(snapshots)
