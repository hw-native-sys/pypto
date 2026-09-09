# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Process-local diagnostic names for assembled callable bytes."""

from pathlib import Path
from typing import Any

from .elf_parser import elf_build_id_64

# ``hid`` (ELF Build-ID 64 of an orchestration ``.so``, lowercase hex) → that
# orchestration's display name (see ``callable_display_name``). The runtime
# identifies a callable in its ``[STRACE]`` timing markers by this hash alone (it
# never emits a name), so recording the pairing at assemble time is what lets
# ``pypto.runtime.benchmark`` label a measured dispatch readably. Process-wide and
# append-only: ``hid`` is content-derived, so two entries can only collide when
# the ``.so`` bytes are identical — in which case the name is identical too.
_CALLABLE_NAMES: dict[str, str] = {}


def callable_display_name(orchestration: dict[str, Any]) -> str:
    """A human-readable, per-program name for an ``ORCHESTRATION`` manifest entry.

    The manifest's ``function_name`` is the fixed AICPU entry symbol the runtime
    dlsym's — ``aicpu_orchestration_entry`` for *every* program — so it cannot
    tell two callables apart. The generated source file is named after the
    orchestration itself (``orchestration/prefill_fwd.cpp``, and for an L3 build
    its ``next_levels/<name>/`` directory matches), so its stem is the
    distinguishing name. Falls back to ``function_name`` if ``source`` is absent.
    """
    source = orchestration.get("source")
    return Path(source).stem if source else str(orchestration.get("function_name", ""))


def register_callable_identity(orch_so: bytes, name: str) -> str:
    """Record ``hid → name`` for an orchestration ``.so`` and return the hid.

    *orch_so* must be the exact buffer handed to the runtime (the same bytes it
    hashes in ``record_device_orch_callable``), so the computed hid matches the
    ``hid=`` field of that callable's ``[STRACE]`` markers.

    Args:
        orch_so: Complete orchestration shared-object bytes.
        name: Display name for the callable — see :func:`callable_display_name`.

    Returns:
        The callable's hid — :func:`~pypto.runtime.elf_parser.elf_build_id_64`
        formatted as lowercase hex, matching the marker wire format.
    """
    hid = f"{elf_build_id_64(orch_so):x}"
    _CALLABLE_NAMES.setdefault(hid, name)
    return hid


def callable_name(hid: str) -> str | None:
    """The orchestration's display name for *hid*, or ``None`` if unknown.

    Returns ``None`` when the callable was not assembled in this process, or on a
    ``*sim`` platform: the sim host seeds the marker hid with the runtime's
    ``callable_id`` rather than the ELF Build-ID, so marker hids do not match the
    hashes recorded here.
    """
    return _CALLABLE_NAMES.get(hid.lower())
