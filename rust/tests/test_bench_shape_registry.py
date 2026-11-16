# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""No corpus may exist for a model whose shapes have no cited source.

This is the gate that keeps the bench meaningful. A plausible-looking middle
configuration measures nothing, and once a corpus is generated from one it is
indistinguishable from a real one -- so the failure is silent and permanent.

The rule is therefore mechanical rather than a matter of care: a shape set is
`sourced` or it is `awaiting_source`, and an `awaiting_source` model must have
an EMPTY corpus. Both directions are checked, because the dangerous one is the
second: a folder quietly filling up with invented shapes.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"
REGISTRY = BENCH / "shapes" / "registry.json"

GENERATED = ("*.tile", "*.pto", "*.cpp", "*.rs", "*.mlir")


@pytest.fixture(scope="module")
def registry():
    assert REGISTRY.is_file(), f"missing {REGISTRY.relative_to(ROOT)}"
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_every_set_declares_a_status(registry):
    assert registry["sets"], "the registry declares no shape sets"
    for name, s in registry["sets"].items():
        assert s.get("status") in ("sourced", "awaiting_source"), (name, s.get("status"))


def test_a_sourced_set_names_its_source_and_carries_values(registry):
    for name, s in registry["sets"].items():
        if s["status"] != "sourced":
            continue
        assert s.get("source"), f"{name} is sourced but names no source"
        assert s.get("values"), f"{name} is sourced but carries no values"
        for k, v in s["values"].items():
            assert v is not None, f"{name}.{k} is null in a sourced set"


def test_an_awaiting_source_set_carries_no_values(registry):
    """The obvious direction, and the cheap one."""
    for name, s in registry["sets"].items():
        if s["status"] != "awaiting_source":
            continue
        assert s.get("values") is None, f"{name} awaits a source but already has values"
        assert s.get("note"), f"{name} awaits a source with no explanation of what is needed"


def test_an_awaiting_source_model_has_an_empty_corpus(registry):
    """The direction that matters: a folder must not quietly fill up.

    If someone generates a corpus before the shapes are decided, this fails and
    names the files -- which is the only point at which the mistake is still
    visible. Afterwards the artefacts look exactly like real ones.
    """
    for name, s in registry["sets"].items():
        if s["status"] != "awaiting_source":
            continue
        folder = BENCH / name
        if not folder.is_dir():
            continue
        found = [str(p.relative_to(BENCH)) for pat in GENERATED for p in folder.rglob(pat)]
        assert not found, (
            f"{name} has no sourced shapes, but its corpus contains generated "
            f"artefacts: {found}. Either record the shape source in "
            f"bench/shapes/registry.json, or remove them -- a corpus built on "
            f"invented shapes cannot be told apart from a real one later."
        )


def test_every_awaiting_source_model_still_has_a_manifest(registry):
    """Empty by design has to be legible as such.

    An absent folder reads as an oversight; a folder with a manifest that says
    what it is waiting for reads as a decision.
    """
    for name, s in registry["sets"].items():
        if s["status"] != "awaiting_source":
            continue
        m = BENCH / name / "manifest.json"
        assert m.is_file(), f"{name} has no manifest saying why it is empty"
        data = json.loads(m.read_text(encoding="utf-8"))
        assert data.get("status") == "awaiting_source", name
        assert data.get("why_empty"), f"{name}'s manifest does not say what it waits on"
        assert data["totals"]["kernels"] == 0, name


def test_a_sourced_model_has_a_non_empty_corpus(registry):
    """The converse. A sourced set with nothing generated is a different kind of
    gap, and worth distinguishing from the one above."""
    for name, s in registry["sets"].items():
        if s["status"] != "sourced":
            continue
        folder = BENCH / f"deepseek_{name}" if not (BENCH / name).is_dir() else BENCH / name
        if not folder.is_dir():
            pytest.skip(f"{name} has no folder yet")
        found = list(folder.rglob("*.tile"))
        assert found, f"{name} is sourced but its corpus is empty"


def test_the_v41_corpus_shapes_match_the_registry(registry):
    """One place for the numbers. The corpus manifest restates the shape set, so
    the two can drift; this is the check that they have not."""
    reg = registry["sets"]["v41"]["values"]
    corpus = json.loads((BENCH / "deepseek_v41" / "manifest.json").read_text(encoding="utf-8"))
    shapes = corpus["shapes"]
    for k, v in reg.items():
        assert shapes.get(k) == v, f"v41.{k}: registry {v} vs corpus {shapes.get(k)}"


def test_the_glm53_corpus_records_what_it_cannot_express():
    """A corpus that omits kernels must say which, and why.

    Two thirds of this model's blocks use a gated delta net, and the tile API has
    no recurrent-scan or sliding-window intrinsic. Approximating either would put
    a kernel in the corpus that computes something the model does not -- so they
    are omitted, and the omission is recorded rather than left for a reader to
    notice from a count.
    """
    m = json.loads((BENCH / "glm_5_3" / "manifest.json").read_text(encoding="utf-8"))
    absent = m["not_in_this_corpus"]
    for name in ("kda_recurrence", "kda_short_conv"):
        assert name in absent, f"{name} is not recorded as absent"
        assert "intrinsic" in absent[name], f"{name} does not name the missing intrinsic"
    assert absent["why_omitted"]
    assert m["totals"]["unexpressible_in_the_tile_api"] == 2


def test_the_glm53_shapes_match_the_registry():
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))["sets"]["glm_5_3"]["values"]
    corpus = json.loads((BENCH / "glm_5_3" / "manifest.json").read_text(encoding="utf-8"))
    for k, v in reg.items():
        assert corpus["shapes"].get(k) == v, f"glm_5_3.{k}: registry {v} vs corpus"


def test_the_generator_refuses_a_shape_set_with_no_source():
    """The generator must not be able to run on an unsourced set.

    Checked on the source text rather than by invoking it, because invoking it
    with a doctored registry would mean writing one -- and the property is that
    the refusal exists at all.
    """
    gen = (BENCH / "shapes" / "gen_glm_5_3.py").read_text(encoding="utf-8")
    assert "refusing to invent any" in gen
    assert "refusing to guess" in gen, "the generator has no guard against a missing dimension"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
