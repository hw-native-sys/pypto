# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A file in `searched/` must be a tiling that was FOUND.

`tile-search` writes its chosen kernel whether or not the choice differs from
the captured value — "chose %cl = 7168, which is the captured value: nothing to
change" still produces a file. Copying that into the corpus claims a rescue
that did not happen, and the claim is invisible: the file is well-formed, it is
in the right directory, and the manifest total goes up by one.

Two such files were committed to `kimi_k3/searched/` before this test existed,
both byte-identical to their input. Both were kernels refused by
`pto-to-ascendc`, and tile-search decides candidates by running the *PTO*
emitter — so a refusal from another emitter is structurally invisible to it. It
answered its own question correctly and the wrong question for those kernels.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"


def corpora():
    return sorted(d for d in BENCH.iterdir() if d.is_dir() and (d / "manifest.json").is_file())


def test_there_are_corpora_to_check():
    assert corpora(), f"no corpus with a manifest under {BENCH}"


@pytest.mark.parametrize("corpus", [d.name for d in corpora()])
def test_no_searched_file_is_a_copy_of_its_input(corpus):
    searched = BENCH / corpus / "searched"
    if not searched.is_dir():
        return
    noops = []
    for f in sorted(searched.glob("*.tile")):
        src = BENCH / corpus / "tile" / f.name
        if src.is_file() and src.read_bytes() == f.read_bytes():
            noops.append(f.name)
    assert not noops, (
        f"{len(noops)} file(s) in {corpus}/searched/ are byte-identical to their input, "
        f"so they claim a tiling that was not found: {noops}\n\n"
        "tile-search writes its choice even when the choice is the captured value. "
        "Check whether the kernel was refused by an emitter tile-search does not "
        "consult -- it decides candidates with the PTO emitter, so an AscendC-side "
        "refusal is invisible to it -- and delete the file rather than keep a copy "
        "of the input."
    )


@pytest.mark.parametrize("corpus", [d.name for d in corpora()])
def test_the_rescue_total_matches_the_directory(corpus):
    man = json.loads((BENCH / corpus / "manifest.json").read_text(encoding="utf-8"))
    claimed = man.get("totals", {}).get("rescued_by_search")
    if claimed is None:
        return
    searched = BENCH / corpus / "searched"
    on_disk = len(list(searched.glob("*.tile"))) if searched.is_dir() else 0
    assert claimed == on_disk, (
        f"{corpus}/manifest.json claims rescued_by_search={claimed} and "
        f"{corpus}/searched/ holds {on_disk} file(s). A total nobody recomputes is "
        "how a manifest comes to describe coverage it does not have."
    )


@pytest.mark.parametrize("corpus", [d.name for d in corpora()])
def test_every_searched_file_names_a_kernel_in_the_corpus(corpus):
    searched = BENCH / corpus / "searched"
    if not searched.is_dir():
        return
    orphans = [
        f.name for f in sorted(searched.glob("*.tile")) if not (BENCH / corpus / "tile" / f.name).is_file()
    ]
    assert not orphans, (
        f"{corpus}/searched/ holds file(s) with no kernel of that name in "
        f"{corpus}/tile/: {orphans}. A searched tiling is a tiling OF something."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_every_searched_row_names_its_oracle_and_its_denominator() -> None:
    """A search result is a MEASUREMENT, so it must say what measured it.

    `tile-search` takes `--oracle pto` (the default) or `ascendc`, and the two
    disagree hard. Over this corpus's five searched rows the PTO oracle accepts 4 to
    7 candidates, while the AscendC oracle accepts **zero** for
    `argmax_v41_8x129280` and `quant_i8_v41_8x5120` -- their refusals are a
    many-row scan and a missing arm, which no choice of extent reaches.

    MEASURED COST OF THE OMISSION, 2026-11-08: a session probed those two under the
    AscendC oracle, read 0 against the recorded 4 and 5, and wrote down that both
    rows were stale. One was. The other reproduces exactly. A row that does not name
    its oracle cannot be re-measured, only re-guessed -- and the wrong guess then
    travelled into two campaign records before anyone re-ran the search.

    `accepted_candidates` needs its denominator for the same reason:
    `argmax_v41_8x131072_pad` is 5 of ELEVEN while the others are out of eight, so a
    bare 5 reads as the same result and is not.

    This asserts the SCHEMA, not the values -- the values are a measurement with a
    date and will move again when an emitter does. What must not happen again is a
    stored number nobody can attribute.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    missing: list[str] = []
    checked = 0
    for manifest in sorted((root / "rust" / "bench").glob("*/manifest.json")):
        corpus = manifest.parent.name
        for stem, entry in json.loads(manifest.read_text()).get("kernels", {}).items():
            searched = entry.get("searched")
            if not isinstance(searched, dict):
                continue
            checked += 1
            where = f"{corpus}/{stem}"
            if searched.get("oracle") not in ("pto", "ascendc"):
                missing.append(f"{where}: no `oracle` field (pto or ascendc)")
            if not isinstance(searched.get("accepted_candidates"), int):
                missing.append(f"{where}: no integer `accepted_candidates`")
            total = searched.get("candidates_total")
            if not isinstance(total, int):
                missing.append(f"{where}: no `candidates_total` -- accepted of how many?")
            elif not 0 <= searched.get("accepted_candidates", -1) <= total:
                missing.append(
                    f"{where}: accepted_candidates {searched.get('accepted_candidates')} "
                    f"is not within 0..{total}"
                )
            if not searched.get("remeasured"):
                missing.append(f"{where}: no `remeasured` date -- when was this last true?")
    assert checked, "no searched row in the tree -- this gate is asleep"
    assert not missing, (
        "a search result that cannot be attributed is a number nobody can re-run:\n  "
        + "\n  ".join(missing)
    )
