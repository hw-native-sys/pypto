# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Partition the bench's refusals into the ones we can fix and the ones we cannot.

A reviewer reading "7 of 20 reach PTO" cannot act on it, because the thirteen
absences are not one thing. Some are this repository's emitters not being
finished. Some are the part refusing a tile that cannot fit in its unified
buffer, where no amount of emitter work helps and the answer is a different
kernel design. Those two belong on different lists, and a single coverage ratio
hides which list a given kernel is on.

So every refusal is classified, and the classification is COMPUTED from the
refusal text the emitters themselves produce rather than typed in beside it.

# The kinds

`EMITTER_GAP`
    `mlir_to_* has no arm for X`. **Ours.** The operation is expressible, the
    emitter has not been taught it, and the emitter says so instead of dropping
    the call silently -- which is the only reason this is a tractable list and
    not a correctness incident. Fixable in this repository.

`HARDWARE_BOUND`
    `UB overflow`, `NPU UB budget`, `NPU tile bound`. **The part's.** The tile
    does not fit, or does not meet the 512-byte fractal stride. Teaching an
    emitter a new arm does not move these; they need a blocked or streamed
    formulation, which is a kernel design decision and not an arm.

`SHAPE_LIMIT`
    The arm **exists** and refuses this *shape*, naming the constraint: a
    `blocked matmul` whose `N` is not a multiple of `Nb`, a packed matvec whose
    pair count is not a multiple of its chunk, an `argmax` over several rows
    where the scan carries one running index. Distinct from `EMITTER_GAP`
    because there is nothing to teach the emitter, and from `HARDWARE_BOUND`
    because the part never saw it. Each message names its own rewrite -- pad the
    extent, or emit one call per row -- so these are closable in the CORPUS
    rather than in the emitter, which is why they are counted apart.

`EMITTER_DEFECT`
    The arm exists, the shape is fine, and the emitter refuses because one of
    **our own passes** does not mark what the arm needs. It names its own
    fix in our code. Currently ZERO rows: the single row it carried until
    2026-11-09 was a malformed call misreported as this kind, now
    `SOURCE_DEFECT`. The kind is kept because the check is still live. This is the most actionable kind in the
    bench and would be invisible folded into any of the others: `EMITTER_GAP`
    would send someone to write an arm that is already written, `SHAPE_LIMIT`
    would send them to pad an extent that is already fine.

`SOURCE_DEFECT`
    The `.tile` source is not a valid program: a call whose argument count
    matches no form its `tile_std` declaration declares. Distinct from
    `SHAPE_LIMIT` because there is no extent to pad and no shape involved at all
    -- folding it there sends someone to re-tile a kernel whose call simply does
    not typecheck. Distinct from `EMITTER_DEFECT` because our code is behaving
    correctly by refusing it.

    Added 2026-11-09 with the check that produces it. Before that check existed
    the same malformed call was read at the longer form's positions, with the
    missing argument defaulting to 0, and surfaced as an `EMITTER_DEFECT` naming
    a pass that had done nothing wrong. One refusal shape moved kinds because the
    emitter learnt to say what was actually wrong.

`CHECKED_LIFT`
    `refused <path>`. The lift behind `ptoas-rs --emit tile` validates more
    strictly than the default one, by design, so it refuses PTO that the PTO
    emitter was willing to produce. A refusal here is a statement about the
    lift's strictness, not about the kernel.

`DERIVED`
    `no PTO to lift`. **Not an independent refusal at all** -- it is the
    downstream shadow of a refusal one stage earlier. Counting it beside its own
    cause would double-count, and reporting it as a separate problem would send
    someone looking for a defect in the lift that is really in the stage before
    it. `reconcile()` asserts each one resolves to a real upstream cause.

# Why an unknown shape FAILS rather than landing in an "other" bucket

A classifier with a catch-all silently absorbs the case it was never taught, and
then its counts look complete while the new refusal shape goes unexamined. So
`classify()` raises on text it does not recognise. Adding a refusal shape to an
emitter is then a change that must also be declared here, which is the point.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent

EMITTER_GAP = "EMITTER_GAP"
EMITTER_DEFECT = "EMITTER_DEFECT"
SHAPE_LIMIT = "SHAPE_LIMIT"
SOURCE_DEFECT = "SOURCE_DEFECT"
HARDWARE_BOUND = "HARDWARE_BOUND"
CHECKED_LIFT = "CHECKED_LIFT"
DERIVED = "DERIVED"

# Where the work lives, which is the question the partition exists to answer.
OURS = (EMITTER_GAP, EMITTER_DEFECT)
THE_CORPUS = (SHAPE_LIMIT, SOURCE_DEFECT)
NOT_OURS = (HARDWARE_BOUND, CHECKED_LIFT)
NOT_A_PROBLEM = (DERIVED,)

# Ordered, and the order matters: the first match wins, so a more specific
# pattern must precede a more general one. Each pattern anchors on wording the
# emitters produce, not on wording a manifest author chose.
PATTERNS = [
    (re.compile(r"^mlir_to_(cpp|pto) has no arm for `(?P<op>[^`]+)`"), EMITTER_GAP),
    # Must precede the SHAPE_LIMIT matmul patterns: this one is also about a
    # blocked matmul, but the thing that must change is our pass, not the shape.
    # A malformed call, caught at the arity check before any positional read.
    # Must precede the deferred-load pattern below: a short call used to reach
    # that check and be reported as it, which is what the kind doc records.
    (
        re.compile(r"^matmul_\w+: call has \d+ argument\(s\)"),
        SOURCE_DEFECT,
    ),
    # Still reachable, and NOT by the malformed call that used to produce it: a
    # well-formed i8 matmul whose operand load is consumed by an intermediate op
    # leaves `detect_blocked_matmul_loads` unable to mark it. The one row this
    # kind carried in the bench was that malformed call, and it was a
    # MISDIAGNOSIS -- the detector requires seven args too, so it had already
    # skipped the line correctly and was being blamed for it.
    (
        re.compile(r"^matmul_i8: both operand loads must be deferred"),
        EMITTER_DEFECT,
    ),
    # An emitter that PRODUCES something invalid is a defect in our code, not a
    # missing arm and not a shape the part refuses. It must precede nothing in
    # particular, but it must exist: the first version of this list had no
    # pattern for it and correctly refused to classify it.
    (re.compile(r"^mlir_to_(pto|cpp) emits INVALID "), EMITTER_DEFECT),
    # A `pto_ir` CONSTRUCTOR refusing an operand pair the emitter itself composed.
    # The arm exists, the corpus shape is fine, and the part never saw it: the
    # emitter built two tiles of different widths and tried to write one into the
    # other. Strictly better than the row above -- that one names PTO already
    # emitted and invalid, this one is caught at construction, where the shape is
    # still attributable to the lowering that chose it -- but the same kind, our
    # own code disagreeing with itself.
    #
    # RESOLVED 2026-11-08, and the hypothesis this comment used to carry was
    # INVERTED. It read: "the emitter chunks -- and one destination was allocated
    # unchunked", labelled "A HYPOTHESIS, not a measurement". Measured, it is the
    # other way round. The allocation WAS narrowed -- `alloc_tile_typed` rewrites
    # cols when they equal the chunked width -- and the operand SPELLINGS were left
    # at the full width, because a chunked retry fixed them up with a text rewrite
    # in `generate_func_pto` that runs AFTER the body is built. A typed `pto_ir`
    # constructor validates at CONSTRUCTION, before that rewrite, so it was shown a
    # source at the chunk width and a destination at the full one. The
    # classification above was right all along: the constructor was correct and the
    # emitter was the defect. Only the direction was wrong, and the hypothesis label
    # is the only reason it did not become a false record.
    #
    # Fixed by narrowing the extent where it ENTERS the emitter
    # (`PtoContext::chunk_cols`, applied in `normalize_tile_call_args` and
    # `translate_scale`), so the two spellings agree before any constructor sees
    # them. kda_conv_glm53_1x8192_k4 now emits at chunk 2048 in 4 blocks and ptoas
    # 0.68 assembles it; dense_gate_up_situ_k3_1x33792 closed with it, which is the
    # evidence that this was one mechanism and not two bugs. The pattern is kept
    # here because the pattern can still fire: any site that builds a type string
    # some other way than through the narrowed extent can reintroduce it.
    #
    # Ruled out by measurement, so the next reader need not repeat it: the same
    # .tile defined `%pj` THREE TIMES, which is invalid SSA and was a real
    # generator defect (gen_glm_5_3.py reused one literal name inside a loop over
    # the taps). Fixed, regenerated, SSA now unique -- AND THE REFUSAL WAS
    # UNCHANGED. The duplicate name was not the cause.
    (re.compile(r"^pto\.\w+: `pto\.\w+` writes "), EMITTER_DEFECT),
    (re.compile(r"^mlir_to_cpp: __tile_\w+ over \d+ rows"), SHAPE_LIMIT),
    (re.compile(r"^blocked matmul: "), SHAPE_LIMIT),
    (re.compile(r"^mul_mv_id_\w+: pair count"), SHAPE_LIMIT),
    (re.compile(r"^(UB overflow|NPU UB budget|NPU tile bound)"), HARDWARE_BOUND),
    (re.compile(r"^no PTO to lift"), DERIVED),
    (re.compile(r"^refused \S+\.pto"), CHECKED_LIFT),
]

REPRESENTATIONS = ("pto", "ascendc", "tile_rs")


def corpora():
    """The corpus folders that carry a manifest, in a stable order."""
    return sorted(p.parent.name for p in BENCH.glob("*/manifest.json"))


def manifest(corpus):
    return json.loads((BENCH / corpus / "manifest.json").read_text(encoding="utf-8"))


def classify(text):
    """`(kind, op_or_None)` for one refusal string.

    Raises on an unrecognised shape. See the module docstring for why that is
    preferred to an `other` bucket.
    """
    for pattern, kind in PATTERNS:
        hit = pattern.match(text)
        if hit:
            op = hit.groupdict().get("op") if kind is EMITTER_GAP else None
            return kind, op
    raise ValueError(
        "unclassified refusal, which means an emitter grew a new refusal shape "
        "and this classifier was not told about it. Add a pattern rather than a "
        f"catch-all:\n  {text[:200]}"
    )


def refusals():
    """Every refusal in the bench as `(corpus, kernel, representation, kind, op, text)`."""
    out = []
    for corpus in corpora():
        kernels = manifest(corpus).get("kernels", {})
        for name, entry in sorted(kernels.items()):
            for rep in REPRESENTATIONS:
                text = entry.get(f"{rep}_refused")
                if not text:
                    continue
                kind, op = classify(text)
                out.append((corpus, name, rep, kind, op, text))
    return out


def reconcile(rows=None):
    """Messages for every `DERIVED` refusal that does not resolve to a cause.

    A `DERIVED` row claims there was no PTO to lift. That is only honest if the
    kernel genuinely has no `.pto` AND something upstream says why. Either half
    failing means the manifest is asserting a consequence whose cause is not
    recorded.
    """
    rows = refusals() if rows is None else rows
    problems = []
    for corpus, name, rep, kind, _, _ in rows:
        if kind is not DERIVED:
            continue
        entry = manifest(corpus)["kernels"][name]
        if entry.get("pto") is not None:
            problems.append(
                f"{corpus}/{name}: {rep} claims there was no PTO to lift, but `pto` is {entry['pto']!r}"
            )
        elif not entry.get("pto_refused"):
            problems.append(
                f"{corpus}/{name}: {rep} claims there was no PTO to lift and "
                "nothing upstream says why -- a consequence with no recorded cause"
            )
    return problems


def missing_arms(rows=None):
    """`{op: sorted [(corpus, kernel, representation)]}` for the EMITTER_GAP rows.

    This is the actionable list: the operations this repository's own emitters
    have not been taught, and every kernel each one costs.
    """
    rows = refusals() if rows is None else rows
    out = {}
    for corpus, name, rep, kind, op, _ in rows:
        if kind is EMITTER_GAP:
            out.setdefault(op, []).append((corpus, name, rep))
    return {op: sorted(v) for op, v in sorted(out.items())}


EMITTER_SOURCE = {
    "mlir_to_pto": HERE.parents[2] / "rust/crates/rspto_emit/src/mlir_to_pto.rs",
    "mlir_to_cpp": HERE.parents[2] / "rust/crates/ascendc_emit/src/mlir_to_cpp.rs",
}


def supports(emitter, op):
    """Does `emitter`'s source actually DISPATCH on `op`?

    Read from the emitter source, NOT inferred from the absence of a refusal.
    An operation that no corpus kernel exercises produces no refusal from
    either emitter, so absence of a refusal is not evidence of support -- and an
    asymmetry built on that inference would invent a supported arm out of a
    corpus gap.

    And it requires a DISPATCH SITE, not merely the name. Both emitters carry a
    list of known intrinsic names, and every operation in this bench appears in
    both lists -- so a check for the bare name reports support for every arm
    either emitter has ever heard of. The documented failure is an emitter
    claiming an operation because only the intrinsic existed, one layer down,
    with nothing lowering it.

    Measured when this was tightened: the loose name test and this one agree on
    all fourteen operation-emitter pairs in the current bench. So the loose form
    was right by luck rather than by construction, and nothing in the report
    moved. It is tightened anyway, because the next operation added to a name
    list without an arm is the one it would have got wrong.
    """
    path = EMITTER_SOURCE[emitter]
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    # The dispatch form both emitters use, excluding doc comments and the
    # in-source MLIR test fixtures, which mention the symbol as `@op`.
    return any(
        f'contains("{op}")' in line or f'contains("@{op}")' in line
        for line in text.splitlines()
        if not line.lstrip().startswith(("///", "//!", "//"))
    )


def asymmetries(rows=None):
    """`{op: (missing_from, present_in)}` where one emitter has the arm and one does not.

    Worth separating from the rest of the gap list, because the semantics are
    already settled by the emitter that has the arm -- so it is the cheapest
    kind of gap to close, and the most embarrassing to leave open.

    Three things this gets right that the obvious version does not. The emitter
    named IN the refusal text is the one MISSING the arm, not the one that has
    it; reading it the other way round reports every row backwards, which is
    what the first version of this function did. The other emitter's support is
    CONFIRMED against its source rather than assumed from its silence. And a
    dispatch site is not support if that emitter REFUSES the shapes the corpus
    holds -- see below.
    """
    rows = refusals() if rows is None else rows
    refused_by = {}
    for _, _, _, kind, op, text in rows:
        if kind is not EMITTER_GAP:
            continue
        emitter = "mlir_to_pto" if text.startswith("mlir_to_pto") else "mlir_to_cpp"
        refused_by.setdefault(op, set()).add(emitter)

    # Every emitter that refuses the operation for ANY reason, not just for a
    # missing arm. `__tile_embedding_f32` is why: mlir_to_cpp dispatches it and
    # then refuses every real vocabulary as a hardware bound (the table cannot
    # be resident), so reading its source alone reports the semantics as settled
    # when the corpus says they are not. An arm that refuses the shape it is
    # said to support is not a settled semantic, so it cannot be the cheap half
    # of a pair.
    refuses = {}
    # A refusal text names its operation only when the operation is MISSING
    # (`has no arm for X`). Other kinds name a shape, a budget or a pass, so
    # attribute them only when the text itself names exactly one of the
    # operations in play. Guessing from the kernel was measured too coarse: it
    # attributed every refusal of a kernel to the one operation another row
    # happened to name, which collapsed five real asymmetries into one.
    known_ops = set(refused_by)
    for _, _, rep, _, op, text in rows:
        emitter = {"pto": "mlir_to_pto", "ascendc": "mlir_to_cpp"}.get(rep)
        if emitter is None:
            continue
        if op is None:
            named = [o for o in known_ops if o in (text or "")]
            op = named[0] if len(named) == 1 else None
        if op is not None:
            refuses.setdefault(op, set()).add(emitter)

    out = {}
    for op, missing in sorted(refused_by.items()):
        others = sorted(set(EMITTER_SOURCE) - missing)
        # An asymmetry needs the OTHER emitter to actually have the arm, AND to
        # accept the shapes this corpus holds.
        present = [e for e in others if supports(e, op) and e not in refuses.get(op, set())]
        if present:
            out[op] = (sorted(missing), present)
    return out


def main() -> int:
    rows = refusals()

    counts = {}
    for _, _, rep, kind, _, _ in rows:
        counts[(rep, kind)] = counts.get((rep, kind), 0) + 1

    print(f"{len(rows)} refusal(s) across {len(corpora())} corpora\n")
    print(f"{'representation':<16}{'kind':<18}count")
    for (rep, kind), n in sorted(counts.items()):
        print(f"{rep:<16}{kind:<18}{n}")

    totals = {}
    for (_, kind), n in counts.items():
        totals[kind] = totals.get(kind, 0) + n
    print()
    for label, kinds in (
        ("work in THIS repository", OURS),
        ("work in the CORPUS", THE_CORPUS),
        ("not ours", NOT_OURS),
        ("not a problem", NOT_A_PROBLEM),
    ):
        total = sum(totals.get(k, 0) for k in kinds)
        parts = ", ".join(f"{k} {totals.get(k, 0)}" for k in kinds)
        print(f"  {label:<26}{total:>4}   ({parts})")
    print(
        "\n`DERIVED` is the shadow of a refusal one stage earlier, so it is not an\n"
        "independent problem. `HARDWARE_BOUND` is the part refusing and\n"
        "`CHECKED_LIFT` is the strict lift doing its job. `SHAPE_LIMIT` closes by\n"
        "changing a corpus extent, not an emitter. Only the first row is code to\n"
        "write here, and `EMITTER_DEFECT` names its own fix."
    )

    arms = missing_arms(rows)
    print(f"\n--- {len(arms)} operation(s) no emitter here has been taught ---")
    for op, where in arms.items():
        print(f"\n  {op}  ({len(where)} kernel-representation pair(s))")
        for corpus, name, rep in where:
            print(f"      {corpus:<14} {rep:<9} {name}")

    asym = asymmetries(rows)
    if asym:
        print("\n--- asymmetries: one emitter has the arm, the other does not ---")
        for op, (missing, present) in asym.items():
            print(
                f"  {op}\n      missing from {', '.join(missing)}; "
                f"present in {', '.join(present)} (confirmed in its source)"
            )
        print(
            "  These are the cheapest gaps to close: the operation's semantics are\n"
            "  already fixed by the emitter that supports it. Support is read from\n"
            "  the other emitter's source, not inferred from it not having refused."
        )

    problems = reconcile(rows)
    if problems:
        print(f"\n{len(problems)} unreconciled derived refusal(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\nEvery derived refusal resolves to a recorded upstream cause.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
