# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""No bench manifest may name an artefact that is not there.

`test_bench_manifest.py` already holds this line for `playground16` and holds it
well -- both directions, totals recomputed rather than asserted. Its scope is the
problem: it names one corpus, so the four MODEL corpora were never gated, and on
2026-11-07 `glm_5_3` was found claiming seven `ascendc/kda_*.cpp` files that exist
in no commit on any branch, each one carrying a `device_check: PASS` line. Seven
kernels read as device-verified against artefacts that never existed.

So this gate DISCOVERS manifests with a glob instead of listing them. Scope was
the defect; a hard-coded list would reproduce it the next time a corpus is added.

Two things are checked, and the second is the one with teeth:

* every per-kernel artefact path resolves to a file, and
* a `device_check: PASS` requires the artefact it passed on.

A path and a hash say an artefact was produced. Only the pairing says the thing
that ran on the device is the thing in the tree, which is the claim a coverage
number actually rests on.

Only `manifest["kernels"]` is walked, deliberately. Top-level bookkeeping records
the opposite of a claim -- `glm_5_3.tile_rs_removed_2026_11_02.files` lists four
`.rs` files that are *correctly* absent -- and a gate that read a removal record
as a claim would demand the restoration of files someone chose to delete.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"

# Extensions that denote an artefact in the tree. A manifest string that ends in
# one of these is a path to be resolved, not prose.
ARTEFACT_EXTS = (".tile", ".pto", ".cpp", ".rs", ".mlir")

# Discovered, not listed -- but the discovery itself is pinned, so that a corpus
# vanishing from disk shows up as a failure here rather than as a gate that
# quietly checks less than it did yesterday.
KNOWN_BENCHES = {"deepseek_v41", "engine", "glm_5_3", "kimi_k3", "playground16"}


def _manifests():
    return sorted(BENCH.glob("*/manifest.json"))


def _claimed_paths(node, out):
    """Every artefact path anywhere under a kernel entry, at any nesting depth.

    Claims live under keys that have multiplied over time -- `ascendc`,
    `ascendc_chunked`, `ascendc_aliased`, `reshaped.pto`, `searched` -- so
    matching on key names would miss the next one. Matching on the VALUE being a
    path cannot.
    """
    if isinstance(node, str):
        if node.endswith(ARTEFACT_EXTS) and " " not in node and "\n" not in node:
            out.add(node)
    elif isinstance(node, dict):
        for value in node.values():
            _claimed_paths(value, out)
    elif isinstance(node, list):
        for value in node:
            _claimed_paths(value, out)


def test_the_gate_sees_every_known_corpus():
    """The scope check, which is the one this file exists because of."""
    found = {m.parent.name for m in _manifests()}
    assert KNOWN_BENCHES <= found, f"a pinned corpus lost its manifest: {sorted(KNOWN_BENCHES - found)}"


@pytest.mark.parametrize("manifest_path", _manifests(), ids=lambda p: p.parent.name)
def test_every_per_kernel_artefact_claim_names_a_file_that_exists(manifest_path):
    bench = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed: set[str] = set()
    _claimed_paths(manifest.get("kernels", {}), claimed)
    assert claimed, f"{bench.name} claims no artefacts at all -- the walk found nothing to check"

    absent = sorted(rel for rel in claimed if not (bench / rel).is_file())
    assert not absent, (
        f"{bench.name}/manifest.json names {len(absent)} artefact(s) that are not in the tree.\n"
        "Regenerate them through the emitter, or correct the manifest at the generator that\n"
        "writes it -- never by hand-editing the manifest to match.\n"
        + "\n".join(f"    {rel}" for rel in absent)
    )


@pytest.mark.parametrize("manifest_path", _manifests(), ids=lambda p: p.parent.name)
def test_a_device_check_pass_requires_the_artefact_it_passed_on(manifest_path):
    """A PASS against a file that does not exist is the defect, stated directly.

    This is not the same assertion as the one above. A kernel could carry a real
    `.cpp` and a `device_check` naming a different one; what makes a device result
    attributable is that the artefact it ran is the artefact in the tree.
    """
    bench = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    unbacked = []
    for name, entry in manifest.get("kernels", {}).items():
        if not isinstance(entry, dict):
            continue
        check = entry.get("device_check")
        if not isinstance(check, str) or not check.strip().upper().startswith("PASS"):
            continue
        # The device route for this corpus is the AscendC artefact.
        artefact = entry.get("ascendc")
        if not isinstance(artefact, str):
            unbacked.append(f"{name}: device_check PASS with no ascendc artefact named")
        elif not (bench / artefact).is_file():
            unbacked.append(f"{name}: device_check PASS but {artefact} is not in the tree")

    assert not unbacked, (
        f"{bench.name}/manifest.json records {len(unbacked)} device PASS line(s) that no\n"
        "artefact in the tree backs. The run cannot be attributed and the kernel cannot be\n"
        "counted as verified.\n" + "\n".join(f"    {row}" for row in unbacked)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
