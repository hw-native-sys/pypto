# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""`verify_all.sh` must cover every kernel whose manifest records a device PASS.

The script opens with "re-run every kernel this tree has verified" and, on
2026-11-05, listed SEVEN while the campaign had twenty-four. It would have printed
"all kernels passed" over a third of the evidence, which is the exact failure the
evidence rules exist to prevent: a claim outliving what backs it.

Nothing compared the two lists, so nothing noticed. This does. A kernel that gains
a `device_check: "PASS ..."` in its manifest and does not appear in the script now
fails here rather than quietly narrowing the reproduction set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"
SCRIPT = ROOT / "rust" / "tools" / "direct-invoke" / "verify_all.sh"


def recorded_passes() -> set[str]:
    """Every artefact the manifests say passed on a device."""
    out: set[str] = set()
    for manifest in sorted(BENCH.glob("*/manifest.json")):
        corpus = manifest.parent.name
        for stem, entry in json.loads(manifest.read_text()).items() if False else json.loads(
            manifest.read_text()
        )["kernels"].items():
            check = entry.get("device_check") or entry.get("arithmetic_check") or ""
            if check.upper().startswith("PASS") or check.upper().startswith("ELEMENT 0 VERIFIED"):
                # the script's table carries the path WITHOUT the extension, because
                # it appends `.cpp` when it builds the command
                out.add(f"{corpus}/ascendc/{stem}")
    return out


def scripted_kernels() -> set[str]:
    """The rows in the script's evidence table."""
    text = SCRIPT.read_text()
    m = re.search(r"ROWS <<'ROWS'\n(.*?)\nROWS", text, re.S)
    assert m, "could not find the ROWS table in verify_all.sh"
    out: set[str] = set()
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        out.add(line.split()[0])
    return out


def test_the_script_covers_every_recorded_device_pass() -> None:
    recorded = recorded_passes()
    scripted = scripted_kernels()
    missing = sorted(recorded - scripted)
    assert not missing, (
        "these kernels record a device PASS but are not in verify_all.sh, so the "
        "script claims to re-run the evidence while skipping them:\n  "
        + "\n  ".join(missing)
    )


def test_the_table_is_the_size_it_looks() -> None:
    """A row count, so the table cannot shrink without someone noticing."""
    recorded = recorded_passes()
    scripted = scripted_kernels()
    assert len(scripted) >= len(recorded), (
        f"{len(scripted)} rows cover {len(recorded)} recorded passes -- the table "
        "has drifted behind the manifests"
    )
