# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Every refusal is classified, and an unknown shape fails rather than hiding.

A coverage ratio says "7 of 20 reach PTO" and a reviewer cannot act on it,
because the thirteen absences are not one thing: some are this repository's
emitters being unfinished, some are the part refusing a tile that does not fit,
and one is the downstream shadow of another. These tests pin that the partition
stays computable, stays total, and keeps pointing at the right owner.

What they do NOT assert is that any particular count is good. The counts move as
the corpus and the emitters move. What must not move is that every refusal lands
in exactly one declared kind, that a derived refusal resolves to a real cause,
and that an asymmetry is confirmed against the other emitter's source rather
than assumed from its silence.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"
sys.path.insert(0, str(BENCH / "shapes"))

import classify_refusals as C  # noqa: E402
from classify_refusals import (  # noqa: E402
    CHECKED_LIFT,
    DERIVED,
    EMITTER_DEFECT,
    EMITTER_GAP,
    EMITTER_SOURCE,
    HARDWARE_BOUND,
    NOT_A_PROBLEM,
    NOT_OURS,
    OURS,
    SHAPE_LIMIT,
    SOURCE_DEFECT,
    THE_CORPUS,
    asymmetries,
    classify,
    corpora,
    manifest,
    missing_arms,
    reconcile,
    refusals,
    supports,
)

# SOURCE_DEFECT added 2026-11-09 with the emitter check that produces it. This
# roster is a REGISTRATION, which `classify_refusals`'s own docstring requires:
# "Adding a refusal shape to an emitter is then a change that must also be
# declared here, which is the point." No assertion below weakens -- the
# partition and exactly-one-kind checks apply to the new kind as to the rest.
KINDS = {
    EMITTER_GAP,
    EMITTER_DEFECT,
    SHAPE_LIMIT,
    SOURCE_DEFECT,
    HARDWARE_BOUND,
    CHECKED_LIFT,
    DERIVED,
}


def test_there_are_refusals_to_classify():
    """Guard against the suite passing because it found nothing.

    A classifier over an empty list satisfies every other test in this file.
    """
    rows = refusals()
    assert len(rows) > 20, f"only {len(rows)} refusals found; the corpus or the reader broke"
    # Five since the engine-slots corpus landed: the four model/playground
    # corpora plus `engine`, whose rows exist for an engine slot rather than a
    # model block. The count is pinned so a corpus cannot appear unclassified.
    assert len(corpora()) == 5


def test_every_refusal_lands_in_exactly_one_declared_kind():
    """Totality. The classifier has no catch-all, so this is also a smoke test.

    `classify()` raises on an unrecognised shape, which means a new refusal
    wording in any emitter fails here instead of being silently bucketed. That
    is deliberate: a classifier with an `other` bucket absorbs exactly the case
    nobody has looked at yet, and its totals keep looking complete.
    """
    for corpus, name, rep, kind, _, text in refusals():
        assert kind in KINDS, f"{corpus}/{name} [{rep}]: {kind} is not a declared kind"
        assert text.strip(), f"{corpus}/{name} [{rep}]: empty refusal text"


def test_an_unknown_refusal_shape_raises():
    """The no-catch-all property, asserted rather than trusted.

    This caught a real omission the first time it ran: the bench carries an
    `argmax over N rows` refusal, which is neither an emitter gap nor a
    hardware bound, and an earlier ad-hoc pass had put it in an `other` bucket
    where nobody would have looked at it.
    """
    with pytest.raises(ValueError, match="unclassified refusal"):
        classify("the emitter declined, for reasons of its own")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("mlir_to_pto has no arm for `__tile_embedding_f32`. Emitting it would", EMITTER_GAP),
        ("mlir_to_cpp has no arm for `__tile_topk_f32`. Emitting it would", EMITTER_GAP),
        ("matmul_i8: both operand loads must be deferred for blocked emit", EMITTER_DEFECT),
        ("blocked matmul: N=16160 must be a multiple of Nb=64", SHAPE_LIMIT),
        ("mul_mv_id_mxfp4_pk: pair count (80) must be a multiple of the 64 chunk", SHAPE_LIMIT),
        ("mlir_to_cpp: __tile_argmax_f32 over 8 rows -- the scan carries", SHAPE_LIMIT),
        ("UB overflow in kernel 'glm53_argmax_154880': buffer 0 needs", HARDWARE_BOUND),
        ("NPU UB budget (arch a2a3): tile of 35840B", HARDWARE_BOUND),
        ("NPU tile bound (arch a2a3): row_major data tile", HARDWARE_BOUND),
        ("no PTO to lift", DERIVED),
        ("refused bench/kimi_k3/pto/rope_k3_1x64.pto", CHECKED_LIFT),
    ],
)
def test_each_kind_has_a_worked_example(text, expected):
    """One sample per kind, so a pattern that stops matching is caught here.

    A screen with no sample of its own hits cannot tell you it went blind -- it
    reports zero and reads as clean.
    """
    kind, _ = classify(text)
    assert kind is expected


def test_the_defect_pattern_wins_over_the_shape_patterns():
    """Order dependence, pinned where it bites.

    The `matmul_i8` refusal is also about a blocked matmul, so a `blocked
    matmul` pattern placed first would swallow it and file a defect in our own
    pass as a shape that needs padding -- sending someone to change a corpus
    extent that is already correct.
    """
    kind, _ = classify("matmul_i8: both operand loads must be deferred for blocked emit")
    assert kind is EMITTER_DEFECT
    kind, _ = classify("blocked matmul: N=16160 must be a multiple of Nb=64")
    assert kind is SHAPE_LIMIT


def test_the_kinds_partition_into_owners_with_nothing_left_over():
    """The four owner buckets must cover every kind exactly once.

    The whole point of the partition is routing. A kind that belongs to no
    bucket, or to two, means a refusal has no owner or two owners, and the
    summary that reads "work in this repository: N" becomes unfalsifiable.
    """
    buckets = list(OURS) + list(THE_CORPUS) + list(NOT_OURS) + list(NOT_A_PROBLEM)
    assert sorted(buckets) == sorted(KINDS)
    assert len(buckets) == len(set(buckets)), "a kind appears in two owner buckets"


def test_every_derived_refusal_resolves_to_a_recorded_cause():
    """A consequence with no cause is a manifest asserting something it cannot show.

    `no PTO to lift` is only honest if the kernel truly has no `.pto` AND the
    PTO stage recorded why. Either half failing means the refusal is unmoored.
    """
    problems = reconcile()
    assert not problems, "\n".join(problems)


def test_a_derived_refusal_whose_cause_is_missing_is_caught(tmp_path, monkeypatch):
    """Prove the reconciliation can fire.

    Seeded by making one kernel claim there was no PTO to lift while carrying a
    `.pto` -- which is the self-contradiction the check exists for.
    """
    real = C.manifest

    def fake(corpus):
        m = real(corpus)
        if corpus == "glm_5_3":
            name = sorted(m["kernels"])[0]
            m["kernels"][name]["tile_rs_refused"] = "no PTO to lift"
            m["kernels"][name]["pto"] = "pto/somewhere.pto"
        return m

    monkeypatch.setattr(C, "manifest", fake)
    problems = C.reconcile()
    assert problems, "a kernel claiming no PTO while holding one was not caught"
    assert "no PTO to lift" in problems[0]


def test_missing_arms_name_an_operation_and_cost_at_least_one_kernel():
    """The actionable list is specific, not a count."""
    arms = missing_arms()
    assert arms, "no emitter gaps found at all, which would be a first"
    for op, where in arms.items():
        assert op.startswith("__tile_"), f"{op} is not an intrinsic name"
        assert where, f"{op} is listed as missing but costs no kernel"
        for corpus, name, rep in where:
            assert corpus in corpora()
            assert name in manifest(corpus)["kernels"]


def _refuses_every_corpus_shape(op):
    """Emitters that DISPATCH `op` and still refuse every corpus kernel using it.

    Read from the refusal rows rather than asserted: an emitter is in this set
    only when the corpus actually contains a kernel it refuses for this
    operation. So an arm that starts accepting a shape leaves the set on its own,
    and nothing has to be un-exempted by hand.
    """
    from classify_refusals import refusals

    out = set()
    for _, _, rep, _, row_op, text in refusals():
        emitter = {"pto": "mlir_to_pto", "ascendc": "mlir_to_cpp"}.get(rep)
        if emitter is None:
            continue
        if row_op == op or (row_op is None and op in (text or "")):
            out.add(emitter)
    return out


def test_an_asymmetry_is_confirmed_against_the_other_emitters_source():
    """Support must be READ, not inferred from the absence of a refusal.

    Two defects this pins, both of which the first version of `asymmetries()`
    had. It reported the emitter named in the refusal text as the one that HAS
    the arm, when that is the one MISSING it -- every row backwards. And it took
    the other emitter's silence as support, when an operation no corpus kernel
    exercises draws no refusal from either emitter.
    """
    for op, (missing, present) in asymmetries().items():
        assert missing and present, f"{op}: an asymmetry needs both sides"
        assert not (set(missing) & set(present)), f"{op}: an emitter cannot be both"
        for emitter in missing:
            # A DISPATCH SITE IS NOT SUPPORT IF THE EMITTER REFUSES EVERY CORPUS
            # SHAPE, which is the rule `asymmetries()` itself already applies on
            # the other side of the pair -- see its `refuses` set and the
            # `__tile_embedding_f32` precedent recorded there: that arm
            # dispatches and then refuses every real vocabulary as a hardware
            # bound, so reading its source alone reports a settled semantic the
            # corpus denies.
            #
            # The same shape arrived from the other direction on 2026-11-09.
            # `mlir_to_cpp` gained a blocked cube arm for
            # `__tile_matmul_i8_acc_i32_dequant_f16` -- recogniser, generator and
            # header-read contract -- and it is deliberately GATED OFF the
            # default route until an i8 oracle scores it, because the uint64
            # packing of the per-column dequant scale is the one part not read
            # from a header. So the source dispatches and every corpus shape
            # still refuses. That is the documented not-support case, not a row
            # the wrong way round, and asserting on the dispatch site alone would
            # force a choice between shipping an unverified emission and
            # deleting a tested arm.
            if emitter in _refuses_every_corpus_shape(op):
                continue
            assert not supports(emitter, op), (
                f"{op} is reported missing from {emitter}, but {emitter}'s source "
                "dispatches on it and does not refuse the corpus shapes -- the "
                "sides are the wrong way round"
            )
        for emitter in present:
            assert supports(emitter, op), (
                f"{op} is reported present in {emitter} without a dispatch site "
                "in its source. Silence is not support."
            )


def test_silence_is_not_support_shown_on_a_case_the_corpus_does_not_have():
    """The property the corpus cannot currently test, tested on a built case.

    `asymmetries()` must confirm the other emitter's support by READING it. A
    version that instead assumed silence meant support produces the IDENTICAL
    report on today's bench, because every operation refused by exactly one
    emitter happens to be genuinely supported by the other, and the one
    operation neither supports is refused by both -- so it never reaches the
    inference. The mutant was undetectable from the corpus.

    That is the trap: the property would have been pinned by the implementation
    and by nothing else, and the next person to simplify the function would get
    a green suite. So the discriminating case is constructed rather than waited
    for: an operation refused by ONE emitter which the OTHER does not support
    either. The honest answer is that this is not an asymmetry at all.
    """
    op = "__tile_mul_mv_id_q8_0"  # dispatched by neither emitter
    assert supports("mlir_to_pto", op) is False
    assert supports("mlir_to_cpp", op) is False

    rows = [
        (
            "glm_5_3",
            "synthetic",
            "pto",
            EMITTER_GAP,
            op,
            f"mlir_to_pto has no arm for `{op}`. Emitting it would drop it.",
        )
    ]
    assert asymmetries(rows) == {}, (
        f"{op} was reported as an asymmetry, but the AscendC emitter does not "
        "support it either. Only one emitter refused it, and that silence was "
        "read as support."
    )


def test_support_requires_a_dispatch_site_not_just_the_name():
    """The name-list trap, asserted against a MEASURED instance.

    Both emitters carry a list of intrinsic names, and in both the list is
    WIDER than the dispatch: measured here, 6 names in the PTO emitter's list
    and 16 in the AscendC emitter's have no dispatch site at all. So a check for
    the bare name would report support for arms that lower nothing, which is the
    documented failure of an emitter claiming an operation because only the
    intrinsic existed one layer down.

    `__tile_mul_mv_id_q8_0` is one such name: present in the PTO emitter's list,
    dispatched by neither emitter. If an arm is ever written for it this test
    fails and wants a different example, which is the correct outcome -- an
    example that stops being an example is not a silent pass.
    """
    listed_only = "__tile_mul_mv_id_q8_0"
    pto = EMITTER_SOURCE["mlir_to_pto"]
    assert f'"{listed_only}",' in pto.read_text(encoding="utf-8"), (
        f"{listed_only} is no longer in the PTO emitter's name list, so this "
        "test has lost the instance that makes the trap real"
    )
    assert supports("mlir_to_pto", listed_only) is False, (
        f"{listed_only} now has a dispatch site; pick another list-only name"
    )

    # The bare `__tile_` prefix guard must not be mistaken for support either.
    # The AscendC emitter routes into its dispatch through `starts_with("__tile_")`,
    # which is a prefix of EVERY intrinsic, so a `starts_with`-aware check would
    # report that emitter as supporting all of them.
    assert supports("mlir_to_cpp", "__tile_attention_decode_f32") is False
    assert supports("mlir_to_pto", "__tile_attention_decode_f32") is False


def test_the_embedding_is_not_an_asymmetry_and_the_corpus_routes_around_it():
    """`__tile_embedding_f32` is not an asymmetry, and the corpus no longer asks.

    It used to be the one operation that ran the other way: present in the
    AscendC emitter and absent from PTO, so the PTO column was short for a
    reason that looked like an unfinished arm. Reading what the AscendC arm
    actually emits shows the "support" was a dispatch site and nothing more: it
    gathers rows WITHIN the weight tile it was handed, while a real vocabulary
    is 2.5-4.7 GB against a 192 KiB unified buffer. So the arm refuses that
    shape, and the operation was refused by BOTH emitters -- PTO for a missing
    GM-driven gather, AscendC for the table not fitting any buffer.

    RESOLVED 2026-11-09, and not the way the previous version of this test
    predicted. It said "if PTO grows the arm, or AscendC grows a GM-driven row
    copy, this test fails and the asymmetry list is where the new fact belongs".
    Neither happened. Neither emitter changed at all: the GENERATORS stopped
    calling the intrinsic, because at decode shape an embedding needs none of
    it. One row copy from GM -- read the index with `__tile_get_value_f32`,
    multiply by the row width, `__tile_load_at_f32` at that element offset --
    and the AscendC emitter already had every step. The only thing missing was a
    DECLARATION for the scalar read-back, which had been implemented and
    undeclared since before this corpus existed.

    So the finding above stands untouched, and what this test now holds is the
    part that can still regress: the intrinsic is still not an asymmetry, no
    corpus kernel routes through a contract that cannot fit, and the refusal the
    embedding kernels DO carry names the one step the PTO emitter genuinely
    lacks.
    """
    op = "__tile_embedding_f32"
    assert op not in asymmetries(), (
        f"{op} is reported as an asymmetry, but the emitter said to support it "
        f"refuses every corpus vocabulary"
    )
    calls = [
        f.relative_to(ROOT)
        for f in sorted(ROOT.glob("rust/bench/*/tile/**/*.tile"))
        if op in f.read_text(encoding="utf-8")
    ]
    assert not calls, (
        f"{op} is called by {len(calls)} corpus kernel(s): {calls}. Its declared "
        f"contract takes the weight table as a TILE, so no arm can satisfy it at "
        f"vocabulary scale -- a kernel routed through it cannot reach the device "
        f"however much static tooling accepts it. The row-copy form is in "
        f"bench/shapes/gen_*.py."
    )
    # INVERTED 2026-11-10. This block asserted that the embeddings carry exactly
    # one refusal -- PTO's missing scalar read-back, an `EMITTER_GAP` naming
    # `__tile_get_value_f32`. That gap is CLOSED, so its subject no longer
    # exists, and the assertion is replaced by its postcondition rather than
    # deleted: the ground it guarded can still regress, and now does so in the
    # other direction.
    #
    # What closed it: `pto.tgetval(src, offset) -> value` had been declared in
    # the ptoas op registry (`_pto_ops_gen.py:22306`) the whole time, and the
    # refusal's wording -- "the one step this emitter cannot take" -- was about
    # the EMITTER but read as if it were about PTO. `mlir_to_pto` now has arms
    # for `__tile_get_value_f32` and `__tile_load_at_f32`; all three embedding
    # `.pto` assemble under `ptoas` and reproduce byte-identically from their
    # own `.tile`.
    #
    # The first two assertions above are UNTOUCHED and still carry the original
    # finding: `__tile_embedding_f32` is not an asymmetry, and no corpus kernel
    # routes through its unfittable contract.
    rows = [r for r in refusals() if r[1].startswith("embedding")]
    assert not rows, (
        f"an embedding refusal reappeared: {rows}. The PTO scalar-read-back gap "
        f"was closed on 2026-11-10 by the `pto.tgetval` arm; a refusal here "
        f"means either that arm regressed or the row-copy form stopped emitting "
        f"on one of the two columns."
    )
    for model, kernel in (
        ("deepseek_v41", "embedding_v41_129280x5120"),
        ("glm_5_3", "embedding_glm53_154880x4096"),
        ("kimi_k3", "embedding_k3_163840x7168"),
    ):
        pto = ROOT / f"rust/bench/{model}/pto/{kernel}.pto"
        assert pto.is_file(), (
            f"{model}/{kernel} has no committed .pto. The embeddings reached the "
            f"PTO column on 2026-11-10 and a missing artefact here is a silent "
            f"withdrawal of that column, which the refusal table cannot see "
            f"because an absent artefact earns no refusal row."
        )
        text = pto.read_text(encoding="utf-8")
        assert "pto.tgetval" in text, (
            f"{model}/{kernel}.pto no longer carries `pto.tgetval`. The scalar "
            f"read-back is the step the kernel exists to exercise; emitting the "
            f"kernel without it would be a silent drop of the index read."
        )


def test_no_corpus_tile_still_uses_the_degenerate_decode_name():
    """All three decode attentions were fixed on 2026-11-10, and this pins it.

    `__tile_attention_decode_f32` used to be the degenerate stub's name: Q, K
    and V all at `1 x cl`, so S=1 and softmax over one score returned V. The
    three now use three different real forms:

      deepseek_v41  `__tile_attention_sink_f32`   sink form, S=8   (ORACLE 53)
      glm_5_3       `__tile_attention_blocked_f32` sourced S=2048 (ORACLE 56)
      kimi_k3       `__tile_attention_dyn_f32`     RUNTIME S       (ORACLE 57)

    The assertion is the absence of the old name from the whole corpus: a
    fixture that goes back to it is the defect this test exists to catch.
    """
    files = sorted(BENCH.glob("*/tile/attention_decode_*.tile"))
    assert files, "no decode-attention fixture in the corpus"
    texts = {p.name: p.read_text(encoding="utf-8") for p in files}
    users = [n for n, t in texts.items() if "__tile_attention_decode_f32" in t]
    assert users == [], (
        "these fixtures are back on the degenerate decode name: "
        f"{users}. See rust/crates/rspto/tests/"
        "the_decode_attention_tiles_are_degenerate.rs for what that name meant."
    )
    assert "__tile_attention_sink_f32" in texts["attention_decode_v41_1x512.tile"]
    assert "__tile_attention_blocked_f32" in texts["attention_decode_glm53_1x256.tile"]
    assert "__tile_attention_dyn_f32" in texts["attention_decode_k3_1x128.tile"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
