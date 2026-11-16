# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared PTOAS synchronization policy, validation and replay metadata."""

import json
from pathlib import Path
from typing import Any, cast

from pypto.pypto_core.backend import BackendType
from pypto.pypto_core.passes import AutoSyncStrategy

_SYNC_FLAGS = {
    AutoSyncStrategy.EVENT: "--enable-insert-sync",
    AutoSyncStrategy.BUFID: "--enable-bufid_sync",
}
_AUTO_FLAGS = {*_SYNC_FLAGS.values(), "--enable-unified-sync", "--enable-inject-barrier-all-sync"}


def sync_flag(strategy: object) -> str:
    """Return exactly one automatic synchronization flag, rejecting unknown policies."""
    if not isinstance(strategy, AutoSyncStrategy):
        raise ValueError(f"sync_strategy must be AutoSyncStrategy.EVENT or BUFID, got {strategy!r}")
    return _SYNC_FLAGS[strategy]


def validate_sync_strategy(strategy: AutoSyncStrategy, backend_type: BackendType) -> None:
    """Reject Buffer-ID synchronization outside A5 before running passes/codegen."""
    sync_flag(strategy)
    if strategy == AutoSyncStrategy.BUFID and backend_type != BackendType.Ascend950:
        raise ValueError(f"sync_strategy=BUFID requires Ascend950 (A5), got {backend_type}")


def validate_sync_flags(strategy: AutoSyncStrategy, flags: list[str]) -> None:
    """Keep handler extras and persisted flags from enabling competing policies."""
    expected = sync_flag(strategy)
    modes = [flag for flag in flags if flag.split("=", 1)[0] in _AUTO_FLAGS]
    if modes != [expected]:
        raise ValueError(
            f"PTOAS synchronization flags are mutually exclusive; sync_strategy={strategy.name} "
            f"requires exactly [{expected!r}], got {modes!r}"
        )
    if strategy == AutoSyncStrategy.BUFID:
        # All normal backend handlers spell the target with these two tokens.
        arches = [flag.split("=", 1)[1] for flag in flags if flag.startswith("--pto-arch=")]
        arches += [flags[i + 1] for i, flag in enumerate(flags[:-1]) if flag == "--pto-arch"]
        if arches != ["a5"]:
            raise ValueError(f"sync_strategy=BUFID requires --pto-arch=a5, got {arches!r}")


def write_sync_metadata(pto_path: Path, strategy: AutoSyncStrategy, flags: list[str]) -> None:
    """Persist the strategy and exact semantic flags per PTOAS compilation unit."""
    validate_sync_flags(strategy, flags)
    pto_path.with_suffix(".json").write_text(
        json.dumps({"sync_strategy": strategy.name.lower(), "ptoas_flags": flags}, indent=2) + "\n",
        encoding="utf-8",
    )


def read_sync_metadata(pto_path: Path, legacy_flags: list[str]) -> list[str]:
    """Restore semantic flags; artifacts without metadata retain legacy event behavior."""
    path = pto_path.with_suffix(".json")
    if not path.exists():
        return legacy_flags
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"expected a JSON object, got {type(raw).__name__}")
        data = cast(dict[str, Any], raw)
        name = data.get("sync_strategy", "event")
        if name not in ("event", "bufid"):
            raise ValueError(f"sync_strategy must be event or bufid, got {name!r}")
        strategy = AutoSyncStrategy.EVENT if name == "event" else AutoSyncStrategy.BUFID
        raw_flags: object = data.get("ptoas_flags", legacy_flags)
        if not isinstance(raw_flags, list):
            raise ValueError(f"ptoas_flags must be a list of strings, got {raw_flags!r}")
        flags: list[str] = []
        for flag in cast(list[object], raw_flags):
            if not isinstance(flag, str):
                raise ValueError(f"ptoas_flags must contain strings, got {flag!r}")
            flags.append(flag)
        validate_sync_flags(strategy, flags)
        return flags
    except ValueError as exc:
        raise ValueError(
            f"Invalid PTOAS compile metadata {path}: {exc}; recompile with ir.compile()"
        ) from exc
