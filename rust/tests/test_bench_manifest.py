# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The bench manifest must describe the files that are actually there.

An index is worth nothing if it can drift from what it indexes. A count that is
written down rather than derived is the usual way a corpus comes to claim
coverage it does not have, so every total in `manifest.json` is recomputed here
from the directory contents.

This is deliberately separate from the byte-identity gate in
`rust/crates/rspto/tests/`, which re-derives the artefacts themselves. That one
needs the built emitter; this one needs nothing, so it runs even on a machine
with no Rust toolchain at all.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench" / "playground16"
MANIFEST = BENCH / "manifest.json"

REPRESENTATIONS = ("tile-rs", "tile", "pto", "ascendc")


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST.is_file(), f"missing {MANIFEST.relative_to(ROOT)}"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_the_manifest_lists_sixteen_kernels(manifest):
    assert len(manifest["kernels"]) == 16
    assert manifest["totals"]["kernels"] == 16


def test_every_named_file_exists(manifest):
    for name, entry in manifest["kernels"].items():
        for rep in REPRESENTATIONS:
            rel = entry.get(rep)
            if rel is None:
                continue
            assert (BENCH / rel).is_file(), f"{name}.{rep} names a missing file: {rel}"
        reshaped = entry.get("reshaped")
        if reshaped:
            for key in REPRESENTATIONS:
                rel = reshaped.get(key)
                if rel is not None:
                    assert (BENCH / rel).is_file(), f"{name} reshaped {key}: {rel}"


def test_every_file_on_disk_is_claimed(manifest):
    """The other direction, which is the one that catches a silent addition."""
    claimed = set()
    for entry in manifest["kernels"].values():
        for rep in REPRESENTATIONS:
            if entry.get(rep):
                claimed.add(entry[rep])
        if entry.get("reshaped"):
            for key in REPRESENTATIONS:
                if entry["reshaped"].get(key):
                    claimed.add(entry["reshaped"][key])

    on_disk = {
        str(p.relative_to(BENCH)) for p in BENCH.rglob("*") if p.is_file() and p.name != "manifest.json"
    }
    assert on_disk == claimed, (
        "the manifest and the corpus disagree.\n"
        f"on disk but unclaimed: {sorted(on_disk - claimed)}\n"
        f"claimed but absent:    {sorted(claimed - on_disk)}"
    )


def test_the_totals_are_recomputed_not_asserted(manifest):
    kernels = manifest["kernels"]
    totals = manifest["totals"]
    assert totals["tile_rs"] == sum(1 for k in kernels.values() if k["tile-rs"])
    assert totals["tile"] == sum(1 for k in kernels.values() if k["tile"])
    assert totals["pto"] == sum(1 for k in kernels.values() if k["pto"])
    assert totals["pto_refused"] == sum(1 for k in kernels.values() if "pto_refused" in k)
    assert totals["reshaped"] == sum(1 for k in kernels.values() if "reshaped" in k)
    assert totals["ascendc"] == sum(1 for k in kernels.values() if k["ascendc"])
    assert totals["ascendc_refused"] == sum(1 for k in kernels.values() if "ascendc_refused" in k)


def test_every_pto_refusal_carries_a_stated_reason(manifest):
    """The set is pinned AND each reason must name its cause.

    Pinning the set is the tripwire: a kernel that silently stops reaching PTO
    shows up here rather than as a column that quietly drops by one.

    Both current members are hardware bounds -- the part refusing a tile. The
    reason-shape assertion also accepts an emitter that produced something
    INVALID, because accepting only "tile bound" or "UB budget" would force an
    emitter defect to be written up as a hardware bound to get past the gate,
    which is a gate shaping the truth to fit itself. The corpus does hold such
    a defect (`pto/fixed/attn_gqa.pto` does not assemble), recorded at the
    corpus level rather than as a kernel refusal because the kernel's own
    capture still emits.
    """
    refused = {n: e["pto_refused"] for n, e in manifest["kernels"].items() if "pto_refused" in e}
    # `gate_up_silu` left this set on 2026-11-05 when rspto gained a chunked path:
    # it emits now, and only `argmax` is still refused at the PTO rung.
    assert set(refused) == {"argmax"}, refused
    for name, reason in refused.items():
        assert reason and reason != "unknown", f"{name}'s refusal has no reason"
        # A reason that does not say what was hit is not a reason. Hardware
        # bounds name the bound; an emitter defect names what it emitted.
        assert any(w in reason for w in ("tile bound", "UB budget", "emits INVALID")), reason


def test_each_refused_kernel_has_a_reshaped_variant(manifest):
    for name in ("argmax",):
        entry = manifest["kernels"][name]
        assert entry.get("reshaped"), f"{name} is refused with no reshaped variant"
        assert entry["reshaped"]["pto"], f"{name}'s reshaped variant has no PTO"


def test_every_kernel_carries_an_ascendc_key(manifest):
    # Present or null, but never omitted: omitting it would make a reader infer
    # the column from a directory listing, and infer it wrong.
    for name, entry in manifest["kernels"].items():
        assert "ascendc" in entry, f"{name} omits the ascendc key"
    assert "ascendc" in manifest["representations"]


def test_the_two_refusals_are_recorded_per_representation(manifest):
    # The emitters check different bounds. They agreed on these two kernels, and
    # recording one verdict for both would hide the day they stop agreeing.
    #
    # `gate_up_silu` LEFT this list on 2026-11-04: the AscendC emitter learned to
    # share slots between dead buffers, so it emits now while PTO still refuses
    # it. That is precisely the disagreement this test was written to surface --
    # it just moved to a different kernel, and `argmax` still shows it.
    # `gate_up_silu` left this set on 2026-11-05: rspto gained a chunked path, so an
    # elementwise chain at 12288 is split into equal chunks and emits. Only `argmax`
    # is still refused, and for a shape bound rather than a budget.
    for name in ("argmax",):
        entry = manifest["kernels"][name]
        assert entry["pto"] is None and entry["ascendc"] is None, name
        assert entry.get("pto_refused"), f"{name} has no PTO reason"
        assert entry.get("ascendc_refused"), f"{name} has no AscendC reason"


def test_the_ascendc_note_says_a_part_must_be_named(manifest):
    # The trap this column has: a blank target has no unified buffer, so every
    # budget check answers Ok and every kernel appears to overflow.
    note = manifest["representations"]["ascendc"]
    assert "--soc" in note


def test_the_corpus_documents_that_tile_rs_is_unverified(manifest):
    # The one claim most likely to be read too generously.
    note = manifest["representations"]["tile-rs"]
    assert "not compiled" in note.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_no_same_type_site_is_still_built_from_strings():
    """The typed migration's finish line, held by a check rather than a claim.

    `rust/tools/count_type_sites.py` parses every `format!` that emits a same-type PTO
    op and counts the sites where two or more independently derived type expressions
    must agree -- the only sites where a typed constructor can refuse something. It
    currently reports zero, which is the definition of done for this migration: what
    remains is either one variable in every slot (nothing to check) or an op whose
    PURPOSE is to change its operands' type.
    """
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    run = subprocess.run(
        ["python3", str(root / "rust/tools/count_type_sites.py")],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    assert "carrying a real check (two or more type expressions must agree): 0" in run.stdout, (
        "a same-type emission site with two derived type expressions is still built from "
        "strings; convert it, or teach the counter why it is not a candidate.\n" + run.stdout
    )
