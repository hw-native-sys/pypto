# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Every committed `.tile` must be what its generator produces today.

This exists because the same mistake was made TWICE in one session: a change to a
generator was committed without the files it produces, so the tree carried a
generator and artefacts that disagreed. Locally nothing looked wrong (the working
tree had the new files); on the remote it was the other way round, and
`check_call_conformance` failed on the pushed commit -- a gate that belongs to
another owner saw it first.

A corpus has three gate owners -- pytest, cargo, the bench scripts -- and a
generator has two outputs, the script and the files it writes. This test is the
one that says so, in the cheapest possible way: regenerate into a scratch tree
and diff.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"
SHAPES = BENCH / "shapes"

GENERATORS = {
    "deepseek_v41": "gen_deepseek_v41.py",
    "glm_5_3": "gen_glm_5_3.py",
    "kimi_k3": "gen_kimi_k3.py",
    # The engine-slots corpus: kernels the Rust serving engine launches, at the
    # shapes it launches them. A row here exists for an ENGINE SLOT rather than a
    # model block, which is why it is a separate corpus and not a model row.
    "engine": "gen_engine.py",
}


def regenerate(tmp: Path, corpus: str, script: str) -> Path:
    """Run one generator in an isolated copy of its own layout.

    The generators write to `<shapes>/../<corpus>/tile`, so the scratch tree
    mirrors that two-level shape rather than passing an output path -- a path
    argument would test a code path the real run never takes.
    """
    shapes = tmp / "bench" / "shapes"
    shapes.mkdir(parents=True)
    shutil.copy(SHAPES / script, shapes / script)
    shutil.copy(SHAPES / "registry.json", shapes / "registry.json")
    (tmp / "bench" / corpus / "tile").mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, str(shapes / script)],
        check=False,
        cwd=shapes,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"{script} refused to run against the committed registry:\n{proc.stderr[-2000:]}"
    )
    return tmp / "bench" / corpus / "tile"


@pytest.mark.parametrize("corpus", sorted(GENERATORS))
def test_the_committed_tiles_are_what_the_generator_produces(corpus: str, tmp_path: Path) -> None:
    produced = regenerate(tmp_path, corpus, GENERATORS[corpus])
    fresh = {p.stem: p.read_text(encoding="utf-8") for p in sorted(produced.glob("*.tile"))}
    assert fresh, f"{GENERATORS[corpus]} produced no tiles"

    committed_dir = BENCH / corpus / "tile"
    committed = {p.stem: p.read_text(encoding="utf-8") for p in sorted(committed_dir.glob("*.tile"))}

    missing = sorted(set(fresh) - set(committed))
    assert not missing, (
        f"{corpus}: {len(missing)} tile(s) the generator writes are NOT committed: "
        f"{missing[:5]}. A generator and its artefacts must land in the same commit."
    )

    differing = [stem for stem in fresh if stem in committed and fresh[stem] != committed[stem]]
    assert not differing, (
        f"{corpus}: {len(differing)} committed tile(s) differ from what the generator "
        f"produces now: {differing[:5]}. Re-run the generator and commit its output."
    )


# Committed tiles that no generator writes and whose provenance the manifest does
# not record. Measured 2026-11-10; this is a CEILING, not an exemption -- the test
# below fails on any tile outside it, so the set can shrink and never grow.
#
# WHY THIS SET WAS INVISIBLE. The reproduce test above compares
# `fresh - committed` (a generator's output must be committed) and the contents of
# the overlap. It never looks at `committed - fresh`, so a tile no generator writes
# is outside every assertion it makes -- and 22 of them are. `glm_5_3` and
# `kimi_k3` are fully generated (27/27 and 28/28); `deepseek_v41`'s generator
# writes 3 of its 17.
#
# NOT A CLAIM THAT THESE WERE HAND-WRITTEN. 7 of deepseek_v41's 14 carry a
# `searched` field in the manifest, which is a recorded third provenance (a shape
# search, with its own oracle note), and those are NOT listed here. The 7 below
# carry no provenance field of any kind: they may be output of a script that was
# never committed, or of an earlier generator. What is true is only that nothing
# in the repository says where they came from, which is what this pins.
#
# It matters because `T` is reported as "a generated .tile exists" and reads
# 72/72. For these kernels the word "generated" is unsupported by anything here.
UNATTRIBUTED_TILES = {
    "deepseek_v41": {
        "gemm_f16_lmhead_v41_16x5120x12928",
        "gemm_i8_v41_16x5120x2304",
        "rmsnorm_v41_1x5120_decode",
        "rmsnorm_v41_8x1280_quarter",
        "rope_v41_8x64",
        "swiglu_v41_1x2304",
        "topk_v41_1x384_k6",
        # These two carry a `searched` KEY whose value is `null`, beside a
        # `search_found_nothing` note saying every candidate tiling was refused
        # because the bound is structural rather than a tile extent. A search
        # that found nothing supplied no provenance, so they belong here.
        #
        # This pair is in the set because the GATE CAUGHT THEM AND THE CENSUS DID
        # NOT. The first count was 7, taken by testing whether the row had a
        # `searched` field; the field is present and holds `null`, so presence
        # said "attributed" and the value says otherwise. Third time in one
        # session that a key-presence test stood in for a usable value -- the
        # others being a truthiness read of `tile-rs: null` and a count of
        # `checks` from the latest record instead of the union.
        "gemm_f16_lmhead_v41_16x5120x16160",
        "gemm_i8_v41_1x5120x2304",
    },
    # The engine corpus is an exploratory probe set, not a scored model, and its
    # eight attention-stage tiles are hand-cut decompositions of one kernel. They
    # are listed rather than exempted for the reason the reproduce test's own
    # comment gives about `engine`: an exemption is how the next corpus goes
    # unwatched.
    "engine": {
        "attn_sink_1head_1blk",
        "attn_sink_engine_64heads",
        "attn_stage1_scores",
        "attn_stage2_softmax",
        "attn_stage2b_softmax_nomax",
        "attn_stage2c_exp_sum",
        "attn_stage2d_exp",
        "attn_stage2e_bcast_den",
    },
}


@pytest.mark.parametrize("corpus", sorted(GENERATORS))
def test_every_committed_tile_is_generated_or_has_a_recorded_provenance(
    corpus: str, tmp_path: Path
) -> None:
    """The direction the reproduce test does not check.

    A tile the generator writes must be committed -- that is checked above. This
    checks the converse: a committed tile must be something the repository can
    account for, either because a generator writes it or because its manifest row
    records another provenance.
    """
    produced = regenerate(tmp_path, corpus, GENERATORS[corpus])
    fresh = {p.stem for p in produced.glob("*.tile")}
    committed = {p.stem for p in (BENCH / corpus / "tile").glob("*.tile")}
    manifest_path = BENCH / corpus / "manifest.json"
    rows = {}
    if manifest_path.is_file():
        rows = json.loads(manifest_path.read_text(encoding="utf-8")).get("kernels") or {}

    def has_provenance(stem: str) -> bool:
        row = rows.get(stem)
        if not isinstance(row, dict):
            return False
        # `searched` is the recorded third route: a shape search, carrying its own
        # chosen shape, candidate count and oracle note.
        return isinstance(row.get("searched"), dict)

    orphans = {s for s in committed - fresh if not has_provenance(s)}
    known = UNATTRIBUTED_TILES.get(corpus, set())

    new = sorted(orphans - known)
    assert not new, (
        f"{corpus}: {len(new)} committed tile(s) are neither generator output nor "
        f"carry a recorded provenance, and are not in UNATTRIBUTED_TILES: {new}. "
        f"A corpus whose premise is that artefacts are products of a generator "
        f"cannot absorb a tile nothing accounts for. Either extend "
        f"{GENERATORS[corpus]} to write it, record a `searched` provenance on its "
        f"row, or add it to the ceiling WITH the reason."
    )
    # The ceiling must stay tight: an entry that no longer applies has to go, or
    # the set stops describing anything. This is what makes it a ratchet.
    stale = sorted(known - orphans)
    assert not stale, (
        f"{corpus}: UNATTRIBUTED_TILES lists {len(stale)} tile(s) that are now "
        f"accounted for: {stale}. Remove them -- a ceiling that outlives its "
        f"members silently licenses the next one."
    )


def test_the_generator_list_covers_every_corpus_with_one() -> None:
    """A new corpus with a generator must be added here, not silently skipped."""
    on_disk = {p.name for p in SHAPES.glob("gen_*.py")}
    listed = set(GENERATORS.values())
    assert on_disk == listed, (
        f"generators on disk {sorted(on_disk)} do not match the ones this test runs "
        f"{sorted(listed)}; add the new corpus rather than letting it go unchecked"
    )


def test_no_corpus_declares_a_generator_it_does_not_have() -> None:
    """And the reverse: a corpus claiming a generator must have one."""
    manifest_names = set()
    for corpus in GENERATORS:
        manifest = BENCH / corpus / "manifest.json"
        if not manifest.is_file():
            continue
        text = json.loads(manifest.read_text(encoding="utf-8"))
        if text.get("corpus"):
            manifest_names.add(corpus)
    assert manifest_names <= set(GENERATORS), (
        f"corpora with a manifest but no generator under test: {sorted(manifest_names - set(GENERATORS))}"
    )


def test_no_generator_uses_one_extent_for_both_store_dimensions() -> None:
    """A store whose two extents are the SAME symbol is latent, not correct.

    Found 2026-11-09 by measurement rather than by reading. The argmax generators
    emit, for an `R x CL` input producing R indices:

        %o = llvm.call @__tile_argmax_f32(%v, %v, %r, %cl)
             llvm.call @__tile_store_i32(%arg1, %o, %r, %r)

    `%r x %r`. At the shapes these generators currently produce, `T = 1`, so that
    is `1 x 1` and **correct by coincidence** -- both kernels are banked and
    device-verified. At `R = 8` the same expression declares 64 elements for 8
    indices, which is exactly what deepseek's two authored argmax tiles do, and
    56 of those 64 are undefined.

    The cause is visible in the generator: the only constants in scope are `%r`
    and `%cl`, with no `%one`, so `%r` was used for both extents. The proof the
    generators CAN express a distinct second extent is their own top-k sibling,
    which emits `(%arg1, %o, %r, %k)`.

    THE RULE THIS PINS, which generalises past argmax: wherever two parameters of
    an expression are EQUAL at the shapes you have, swapping them is undetectable
    -- so a passing test says nothing about which one belongs where. This asserts
    on the SOURCE rather than on the output, because the output is where the two
    readings agree.

    THE DISCRIMINATION THAT MAKES THIS A GATE AND NOT NOISE, found by running the
    first version: a repeated extent symbol is fine when the symbol is bound to
    the LITERAL 1, and latent when it is bound to a VARIABLE that happens to be 1.

    `c("cr1", 1)` then `store(..., %cr1, %cr1)` is a declared 1x1 scalar result
    -- the symbol is named for its value and cannot be anything else, so the
    repetition carries the meaning "one element". That is six sites across two
    generators and all six are correct.

    `c("r", T)` then `store(..., %r, %r)` is the defect: `%r` is the ROW COUNT,
    which is 1 only because these corpora are single-batch. The first version of
    this test flagged both forms and fired on seven sites, of which one was the
    real one -- a gate that cannot tell them apart would be turned off.
    """
    import re

    offenders = []
    for corpus, script in sorted(GENERATORS.items()):
        path = SHAPES / script
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        # Symbols bound to the literal 1 anywhere in the generator. A symbol
        # bound to 1 in one place and to something else elsewhere is NOT in this
        # set, which is the conservative direction.
        # Symbols EVERY one of whose bindings is the literal int 1. A symbol
        # bound to 1 in one place and to an expression elsewhere is NOT in this
        # set, which is the conservative direction.
        bindings: dict[str, set[str]] = {}
        for m in re.finditer(r'c\(\s*"(\w+)"\s*,\s*([^),]*)', text):
            bindings.setdefault(m.group(1), set()).add(m.group(2).strip())
        literal_one = {s for s, vals in bindings.items() if vals == {"1"}}
        for n, line in enumerate(text.splitlines(), 1):
            if "@__tile_store_" not in line:
                continue
            m = re.search(r"@__tile_store_\w+\(\s*(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\s*\)", line)
            if not m or m.group(3) != m.group(4):
                continue
            sym = m.group(3).lstrip("%")
            if sym in literal_one:
                continue  # a declared scalar, named for its value
            offenders.append(
                f"{script}:{n}: both store extents are {m.group(3)}, which is not bound to the literal 1"
            )

    assert not offenders, (
        "a generator emits a store whose two extents are the same symbol. At the "
        "shapes it produces today this may be correct by coincidence (R=1 makes "
        "`%r x %r` equal `1 x 1`), and it declares R*R elements for R results at "
        "any other R. Give the second extent its own constant:\n  "
        + "\n  ".join(offenders)
    )
