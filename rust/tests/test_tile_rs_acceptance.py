# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep the tile-rs acceptance backlog honest.

`rust/bench/tile_rs_acceptance.json` lists every declared kernel that NEITHER
emitter places, with the reason each gave. It exists so that a kernel the AscendC
and PTO paths both refuse stays a TARGET -- something the lift can be taught to
take -- instead of quietly dropping out of the inventory.

The file is only worth having if it is true, and it goes stale in both directions:

  * a kernel is refused and not listed  -> the backlog understates the work;
  * a kernel is listed and starts placing -> the file describes a problem that no
    longer exists, which is the exact failure mode that made the README claim seven
    kernels when there were twenty-four.

So this test pins both directions against the manifests.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"
BACKLOG = BENCH / "tile_rs_acceptance.json"
MODELS = ("deepseek_v41", "glm_5_3", "kimi_k3")


def unplaced() -> set[str]:
    """Kernels with neither representation, read from the manifests."""
    out: set[str] = set()
    for model in MODELS:
        manifest = json.loads((BENCH / model / "manifest.json").read_text())
        for stem, entry in manifest["kernels"].items():
            if not entry.get("pto") and not entry.get("ascendc"):
                out.add(f"{model}/{stem}")
    return out


def listed() -> dict[str, dict]:
    return json.loads(BACKLOG.read_text())["kernels"]


def test_every_unplaced_kernel_is_in_the_backlog() -> None:
    missing = sorted(unplaced() - set(listed()))
    assert not missing, (
        "these kernels are refused by both emitters and are not in "
        "tile_rs_acceptance.json, so the backlog understates what tile-rs has to "
        "accept:\n  " + "\n  ".join(missing)
    )


def test_the_backlog_holds_no_kernel_that_now_places() -> None:
    stale = sorted(set(listed()) - unplaced())
    assert not stale, (
        "these kernels are in the acceptance backlog but at least one emitter now "
        "places them -- remove them, or the file describes work that is done:\n  "
        + "\n  ".join(stale)
    )


def test_every_entry_says_why_each_representation_refused() -> None:
    thin = [
        name
        for name, rec in listed().items()
        if len((rec.get("pto") or "").strip()) < 10
        or len((rec.get("ascendc") or "").strip()) < 10
    ]
    assert not thin, (
        "an entry without both reasons is a name in a list, not a target:\n  "
        + "\n  ".join(sorted(thin))
    )
