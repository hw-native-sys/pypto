# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The per-corpus kernel estimate is recomputed here, never read.

A denominator is the easiest number in a repository to let rot. The corpora grow
and shrink, the registry gains a dimension, and a figure typed into a manifest
keeps reading as current while the ratio it anchors drifts. So every assertion
below recomputes the estimate from the registry and compares it to what the
manifest claims.

What it does NOT assert is that the estimate is *right*. It is a declared
inventory of a decode step, and whether a decode step needs a separate kernel
for `k_proj` and `v_proj` is a judgement. The test pins that the number in the
manifest is the number the rules produce, and that the rules keep saying which
rows they could not count.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"
sys.path.insert(0, str(BENCH / "shapes"))

from estimate_kernels import (  # noqa: E402
    GROUND_TRUTH,
    REGISTRY_COUNTS,
    REGISTRY_GENERATED_ON,
    REGISTRY_IS_REDACTED,
    REGISTRY_PLAYGROUND16,
    REGISTRY_READ_ON,
    counted,
    estimate,
    inventory,
    sets,
)


def manifest(corpus):
    return json.loads((BENCH / corpus / "manifest.json").read_text(encoding="utf-8"))


def test_there_are_corpora_with_an_estimate():
    have = [c for c in sets() if (BENCH / c / "manifest.json").is_file()]
    assert len(have) >= 4, f"expected four corpora with manifests, found {have}"


@pytest.mark.parametrize("corpus", sorted(sets()))
def test_the_denominator_is_never_below_the_numerator(corpus):
    """The invariant, and the one this file exists for.

    `playground16` derived to 13 against a corpus of 16, and that was reported
    as the corpus "exceeding its estimate" -- which reads as a quirk and was a
    defect. A model predicting fewer kernels than a corpus already holds has
    been refuted by that corpus. The fix was not a note; it was to take the
    known count of sixteen.
    """
    block = manifest(corpus)["estimated_decode_kernels"]
    have = counted(corpus)["tile"]
    assert block["count"] >= have, (
        f"{corpus}: estimate {block['count']} is below the {have} kernels the "
        "corpus already holds. The corpus has refuted the estimate."
    )


@pytest.mark.parametrize("corpus", sorted(sets()))
def test_the_manifest_matches_what_the_rules_produce(corpus):
    man = manifest(corpus)
    block = man.get("estimated_decode_kernels")
    assert block, f"{corpus}/manifest.json carries no estimated_decode_kernels"
    count, floor, _ = estimate(corpus, sets()[corpus])
    assert block["count"] == count, (
        f"{corpus} claims {block['count']} and the rules produce {count}. A "
        "denominator nobody recomputes is how a coverage ratio comes to describe "
        "a corpus that changed underneath it."
    )
    assert block["is_lower_bound"] == floor


@pytest.mark.parametrize("corpus", sorted(GROUND_TRUTH))
def test_a_known_count_beats_the_derivation(corpus):
    """Where a true count exists it is used, and the derivation is kept beside
    it rather than discarded -- the gap between them is the only measurement of
    how coarse the derivation is."""
    block = manifest(corpus)["estimated_decode_kernels"]
    known, _, _ = GROUND_TRUTH[corpus]
    assert block["count"] == known
    inv, _ = inventory(sets()[corpus])
    assert block["derived_inventory_total"] == sum(n for _, n in inv)
    if block["derived_inventory_total"] != known:
        assert "inventory_is_not_used_here" in block, (
            f"{corpus}'s derivation and its known count differ and the manifest "
            "does not say which one the count is"
        )


@pytest.mark.parametrize("corpus", sorted(sets()))
def test_the_collapses_are_named_not_just_counted(corpus):
    """A ratio is not actionable; a list of omissions is.

    "under-counts by 3.4x" tells a reader nothing they can correct. Each
    manifest names the specific rows that stand for several real kernels -- the
    expert gate/up/down triple, the attention variants, the per-dtype and
    per-quantisation matvecs, the absent prefill shapes -- so someone holding
    the true count can see which rows to expand.
    """
    block = manifest(corpus)["estimated_decode_kernels"]
    collapses = block.get("what_the_inventory_collapses")
    assert collapses and len(collapses) >= 5, (
        f"{corpus} records the derivation error as a ratio without naming what "
        "the inventory collapses, which is the part a reader can act on"
    )
    joined = " ".join(collapses).lower()
    for expected in ("expert", "attention", "matvec", "prefill"):
        assert expected in joined, (
            f"{corpus}'s collapse list does not mention {expected}, which is one of the known ones"
        )


def test_the_derivation_records_how_wrong_it_is():
    """One anchor, and it says so rather than being quietly corrected.

    Multiplying every derived figure by the one measured ratio would produce
    numbers with the appearance of measurement and none of the substance, so
    the ratio is recorded and not applied.

    The ratio is RECOMPUTED here rather than grepped for. An earlier version of
    this test asserted the literal string "3.4" appeared in the generator,
    which is satisfied by the digits occurring anywhere -- including in a
    sentence that had gone stale against the two numbers it describes. It did
    in fact go stale, when the anchor moved from 51 to the registry's 52.
    """
    anchor = "deepseek_v41"
    known, _, _ = GROUND_TRUTH[anchor]
    rows, _ = inventory(sets()[anchor])
    predicted = sum(k for _, k in rows)
    ratio = known / predicted
    stated = f"{ratio:.1f}x"

    gen = (BENCH / "shapes" / "estimate_kernels.py").read_text(encoding="utf-8")
    assert "Not applied as a correction" in gen
    assert stated in gen, (
        f"the inventory predicts {predicted} for the anchor against a known "
        f"{known}, a ratio of {stated}, and that spelling does not appear in "
        "the generator. The prose has drifted from the two numbers it describes."
    )
    assert str(predicted) in gen and str(known) in gen
    for corpus in sets():
        block = manifest(corpus)["estimated_decode_kernels"]
        assert "under-counts" in block["derivation_error"]
        assert stated in block["derivation_error"]


def test_the_corpus_and_the_published_registry_name_the_same_sixteen():
    """Agreement that is worth something, because the two were built apart.

    The baseline corpus was assembled from one tree in November; the published
    registry's paper collection was generated from another at the end of
    August. Neither was copied from the other, so set equality between them is
    independent corroboration of the sixteen -- which is the one denominator in
    this bench that is a definition rather than an estimate, and so the one
    that nothing else could check.

    Asserted as set equality in BOTH directions. A subset check in one
    direction would pass while the corpus quietly carried a seventeenth kernel,
    and in the other while it was missing one.
    """
    registry = set(REGISTRY_PLAYGROUND16)
    assert len(registry) == 16, "the recorded collection is not sixteen names"
    corpus = {p.stem for p in (BENCH / "playground16" / "tile").glob("*.tile")}
    assert corpus == registry, (
        f"only in the corpus: {sorted(corpus - registry)}; only in the registry: {sorted(registry - corpus)}"
    )


def test_the_registry_is_recorded_as_dated_and_partial():
    """A floor from an external source, with the reasons it is a floor.

    The published view is marked redacted and is stamped two months before this
    corpus was assembled. Both facts are what make its counts floors rather
    than figures, so both are carried next to the counts instead of being
    remembered.
    """
    assert REGISTRY_IS_REDACTED, (
        "the published view is marked redacted; if that ever stops being true "
        "the counts stop being floors and this file's reasoning changes"
    )
    assert REGISTRY_GENERATED_ON < REGISTRY_READ_ON, (
        "the registry was read before it was generated, which cannot be right"
    )
    gen = (BENCH / "shapes" / "estimate_kernels.py").read_text(encoding="utf-8")
    assert "redacted" in gen and REGISTRY_GENERATED_ON in gen


def test_the_registry_does_not_override_a_stronger_derivation():
    """The asymmetry is deliberate, and this pins it.

    An external count wins where it is LARGER than the derivation, because both
    are floors and the larger floor is the better one. Where it is smaller it
    loses, because adopting it would lower a bound on the strength of its
    provenance alone. The registry's GLM 5.3 row is the case that distinguishes
    the two, and without this test the next person to see a registry number
    would have no way to tell which rule produced the manifest.
    """
    for corpus, registry_count in REGISTRY_COUNTS.items():
        derived, _, _ = estimate(corpus, sets()[corpus])
        if corpus in GROUND_TRUTH:
            known, _, source = GROUND_TRUTH[corpus]
            assert known >= registry_count, (
                f"{corpus}: the known count {known} is below the registry's "
                f"{registry_count}, so the larger floor was discarded"
            )
            assert "registry" in source, f"{corpus} takes the registry's count but does not say so"
        else:
            assert registry_count <= derived, (
                f"{corpus}: the registry says {registry_count} and the "
                f"derivation says {derived}. The registry is the larger floor "
                "and is being ignored -- adopt it or record why not."
            )

    assert "kimi_k3" not in REGISTRY_COUNTS, (
        "the registry lists no Kimi K3, and a 0 recorded here would read as a "
        "measurement of zero rather than as an absence"
    )


@pytest.mark.parametrize("corpus", sorted(sets()))
def test_a_lower_bound_says_it_is_one(corpus):
    block = manifest(corpus)["estimated_decode_kernels"]
    if not block["is_lower_bound"]:
        return
    assert "LOWER BOUND" in block["source"] or "floor" in block["source"], (
        f"{corpus}'s count is a lower bound and its source does not say so, so it will be read as the figure"
    )


@pytest.mark.parametrize("corpus", sorted(sets()))
def test_the_inventory_sums_to_the_derived_total(corpus):
    block = manifest(corpus)["estimated_decode_kernels"]
    assert sum(r["n"] for r in block["inventory"]) == block["derived_inventory_total"], (
        f"{corpus}'s inventory does not sum to its own derived total, so one of the two was edited by hand"
    )


@pytest.mark.parametrize("corpus", sorted(sets()))
def test_named_gaps_reconcile_against_the_derived_total(corpus):
    """Nothing is silently absent.

    The gaps were enumerated against the INVENTORY, so they reconcile against
    the derived total -- not against a known count that came from elsewhere.
    Conflating the two would make the arithmetic look broken when it is the
    denominators that differ.
    """
    block = manifest(corpus)["estimated_decode_kernels"]
    gaps = block.get("rows_with_no_kernel")
    if not gaps:
        return
    total = 0
    for key, text in gaps.items():
        head = text.split(" row", 1)[0].strip()
        assert head.isdigit(), f"{corpus}'s gap `{key}` has no leading row count"
        total += int(head)
    have = counted(corpus)["tile"]
    assert have + total == block["derived_inventory_total"], (
        f"{corpus}: {have} present + {total} named as gaps = {have + total}, "
        f"against a derived inventory of {block['derived_inventory_total']}. The "
        "difference is a row neither expressed nor explained."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
