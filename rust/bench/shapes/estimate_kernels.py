# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""How many distinct kernels one decode step of each model needs.

A corpus count on its own says nothing. "11 kernels" for Kimi K3 is either most
of the model or a fifth of it, and the manifests could not tell you which. This
supplies the denominator.

# Ground truth first, derivation only as a fallback

Two of the four corpora have a count that is KNOWN rather than derived:

| model | kernels | source |
|---|---|---|
| DeepSeek-R1-Qwen-Distilled | **16** | the baseline set, defined as sixteen; the registry agrees |
| DeepSeek V4 Flash | **52** | the published kernel registry, read 2026-11-02 |

V4.1 Flash is **at least** 52: it is a later revision of the same architecture,
so the count can only grow. It is recorded as a lower bound, not as 52.

The registry row is itself a floor three times over. Its published view is
marked redacted, so it is a subset. 39 of the 52 rows carry a `source-scan`
basis rather than a `trace`, and the registry's own wording is that only a
trace is evidence a kernel holds measurable time. And it was generated two
months before this corpus was assembled.

These take precedence over anything derived. For GLM 5.3 and Kimi K3,
`inventory()` derives a figure from the model's own configuration, and that
figure is a **LOWER BOUND, not an estimate of the true count**. The registry
does not override either of them, for two different reasons: it lists Kimi K3
not at all, and its six GLM 5.3 rows are FEWER than the derivation, under an
unverified basis dated before that architecture's work landed. A weaker
external figure is not provenance for replacing a stronger derived one.

# ⛔ The derivation under-counts by about 3.5x, measured

The inventory predicted **15** for DeepSeek V4 Flash against a registry count
of **52**. So the rules below under-count by roughly **3.5x** against the one
model where both numbers exist, and the reason is visible in the rows: the
inventory collapses into a single line things that are many kernels in a real
engine -- each expert's gate, up and down as one row; one attention core rather
than its decode, prefill and partial variants; one matvec rather than the f16,
f32 and per-quantisation forms; and no prefill shapes at all.

That ratio is NOT applied as a correction factor. One anchor cannot calibrate a
model, and multiplying a wrong derivation by 3.5 would produce a number with the
appearance of measurement and none of the substance. The derived figures are
labelled lower bounds and left alone.

# The invariant that caught this

**A denominator below the numerator is wrong.** `playground16` derived to 13
against a corpus of 16, and that was reported as the corpus "exceeding its
estimate" -- which reads as a quirk and was a defect. A model that predicts
fewer kernels than a corpus already holds has been refuted by that corpus.
`check()` enforces it, and `rust/tests/test_bench_kernel_estimate.py` asserts it
per corpus.

# What a "kernel" is counted as

One distinct (operation, shape) pair, because that is what the corpus holds: a
file named `rmsnorm_glm53_1x4096` serves all 46 of that model's blocks, since
they share `D`. A layer count does not multiply the figure.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "registry.json"
BENCH = HERE.parent

# Shapes for the playground corpus's model, which has no registry entry: it is
# the hand-written baseline rather than a generated set. Read from
# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B's own config.json on 2026-11-02 and
# confirmed against the corpus -- D 1536 is the extent its residual_add and
# rms_norm kernels carry.
PLAYGROUND = {
    "architecture": "qwen2",
    "D": 1536,
    "DENSE_FFN_INTER": 8960,
    "VOCAB": 151936,
    "HEAD_COUNT": 12,
    "KV_HEAD_COUNT": 2,
    "ROPE": 1,  # rope_theta present
    "N_EXP": 0,  # dense
    "BLOCKS": 28,
}


# The published kernel registry, and what it is allowed to settle.
#
# The project publishes a kernel-to-model dependency registry as a web
# application. Its data is a single JSON object carrying `devices`, `kernels`,
# `models`, `model_kernels`, `variants` and `measurements`; it is marked
# `redacted: true` and stamped `generated_on: 2026-08-31`. Those two fields are
# why it is read as EVIDENCE WITH A DATE and not as a definition:
#
#   * `redacted` means the published view is a subset of the real one, so every
#     count it gives is a FLOOR and never a ceiling.
#   * `generated_on` is two months before this corpus was assembled, so a model
#     whose work landed later is under-represented by construction. Two of its
#     ten models carry ZERO kernel rows, which is the registry demonstrating
#     that absence in it does not mean absence in fact.
#
# Each `model_kernels` row carries a `basis` saying how the row was obtained:
# `trace` (observed in a run), `source-scan` (read out of an engine's sources)
# or `doc`. A model with only `source-scan` rows has never been traced, and the
# registry says elsewhere that only a trace is evidence a kernel holds
# measurable time.
#
# Read on 2026-11-02 from the published application, counting distinct
# `kernel_id` per `model_id`:
#
#   DeepSeek-V4-Flash               52   13 trace + 39 source-scan
#   Gemma 4 26B A4B                 25   source-scan
#   Qwen 3.8 27B                    24   source-scan
#   Poolside Laguna XS 2.1          22   doc
#   Qwen3.8-Flash-Next              21   source-scan
#   DeepSeek-R1-Distill-Qwen-1.5B   20   trace
#   Qwen3-4B                        11   trace
#   GLM-5.3                          6   source-scan
#   GLM-5.2                          0
#   Qwen3.6-35B-A3B                  0
#
# It also carries named collections, one of which is the sixteen-kernel paper
# corpus this bench's `playground16` was built from.
REGISTRY_READ_ON = "2026-11-02"
REGISTRY_GENERATED_ON = "2026-08-31"
REGISTRY_IS_REDACTED = True

# The registry's own collection of the sixteen kernels, verbatim. This is here
# to be COMPARED against the corpus rather than to describe it: the corpus was
# assembled from a different tree, months after the registry was generated, so
# agreement between the two is worth something and a list copied out of the
# corpus itself would be worth nothing.
REGISTRY_PLAYGROUND16 = (
    "argmax",
    "attn_gqa",
    "down_proj",
    "gate_up_silu",
    "k_proj",
    "matmul",
    "o_proj",
    "q_proj",
    "residual_add",
    "rms_norm",
    "rope",
    "v_proj",
    "vec_add",
    "vec_exp",
    "vec_mul",
    "vec_sub",
)

# Distinct kernels per model in the published registry, for the models this
# bench covers. A floor, for the two reasons above.
REGISTRY_COUNTS = {
    "deepseek_v41": 52,
    "glm_5_3": 6,
    # kimi_k3 is deliberately absent: the registry lists ten models and none of
    # them is Kimi K3. Writing a 0 here would be read as a measurement.
}


# Counts that are KNOWN rather than derived. These win over `inventory()`.
#
# Three of the four corpora resolve differently, and the differences are the
# point -- a single rule applied to all four would have had to be wrong about
# three of them.
#
# `playground16`  Sixteen by DEFINITION, and now also by agreement: the
#                 registry's paper collection and this corpus hold the same
#                 sixteen names, with no extras on either side.
#
# `deepseek_v41`  A floor of 52, from the registry's DeepSeek V4 Flash row.
#                 This supersedes an earlier floor of 51 taken from the
#                 owner's count; 52 is the larger of the two and is itself a
#                 floor, since 39 of the 52 are `source-scan` and the published
#                 view is redacted. V4.1 is a later revision of the same
#                 architecture, so its count can only grow from here.
#
# `glm_5_3`       NOT taken from the registry. The registry's six rows are all
#                 model-prefixed kernels under a `source-scan` basis dated
#                 before this architecture's work landed, and six is SMALLER
#                 than what this bench derives from the model's own
#                 configuration. Replacing a derivation with a weaker external
#                 figure would be a regression dressed as provenance, so the
#                 derived lower bound stands and the registry's six is recorded
#                 above as the floor on model-specific kernels alone.
#
# `kimi_k3`       The registry does not list this model at all, so it has
#                 nothing to say and the derivation stands unchallenged.
GROUND_TRUTH = {
    "playground16": (
        16,
        False,
        "the playground baseline set, defined as sixteen kernels, and the "
        "published registry's paper collection names the same sixteen",
    ),
    "deepseek_v41": (
        52,
        True,
        "the published kernel registry's DeepSeek V4 Flash row, read "
        f"{REGISTRY_READ_ON}: 52 distinct kernels, 13 of them traced and 39 "
        "read from sources. A floor three times over -- the view is redacted, "
        "most rows are unverified by a trace, and V4.1 is a later revision of "
        "the same architecture",
    ),
}

# The one place both a derived figure and a true count exist, so the only
# measurement of how wrong the derivation is.
DERIVATION_ERROR = (
    "the inventory predicted 15 for DeepSeek V4 Flash against a registry count "
    "of 52, so it under-counts by about 3.5x. Not applied as a correction: one "
    "anchor cannot calibrate a model, and a derivation multiplied by 3.5 would "
    "look measured and would not be."
)


# What the inventory COLLAPSES, named rather than summarised as a ratio.
#
# "under-counts by 3.5x" is one number and tells a reader nothing they can act
# on. These are the specific places a row stands for several real kernels, each
# with the multiplier it hides. They are NOT multiplied out -- doing that would
# turn a list of known omissions into a fabricated total -- but they are listed,
# so someone who knows the true count for a model can see which rows to expand.
COLLAPSES = [
    (
        "each expert's gate, up and down are one row",
        "a real engine has three kernels there, and the gate/up pair is often "
        "fused separately from the down-projection",
    ),
    (
        "one attention core",
        "decode, prefill and partial/chunked are different kernels with "
        "different shapes, and a model with two attention types doubles that "
        "again",
    ),
    (
        "one matvec per projection",
        "f16 and f32 weight forms are separate kernels, and each quantisation "
        "the engine supports adds another (q8_0, q4_K, iq2_xxs, mxfp4 ...)",
    ),
    (
        "no prefill shapes",
        "every projection and norm has a batched prefill form, which is a "
        "different kernel from its decode form",
    ),
    (
        "no quantise/dequantise kernels",
        "activation quantisation and weight unpacking are real kernels; the "
        "inventory treats weight format as out of scope",
    ),
    (
        "one sampling kernel",
        "argmax is counted; top-p, top-k sampling and draft verification are not",
    ),
]


def estimate(corpus, shapes):
    """`(count, is_lower_bound, source)` for one corpus."""
    if corpus in GROUND_TRUTH:
        n, floor, src = GROUND_TRUTH[corpus]
        return n, floor, src
    inv, unknown = inventory(shapes)
    n = sum(k for _, k in inv)
    return (
        n,
        True,
        "derived from the model's own configuration by inventory(); a LOWER "
        "BOUND, because " + DERIVATION_ERROR,
    )


def check(corpus, count, have):
    """A denominator below the numerator is a refuted model, not a quirk."""
    if count < have:
        raise AssertionError(
            f"{corpus}: the estimate is {count} and the corpus already holds "
            f"{have} kernels. A denominator below the numerator is wrong -- the "
            "corpus has refuted the model. Fix the estimate, do not report the "
            "corpus as exceeding it."
        )


def inventory(s):
    """`([(line, n)], [unknown])` -- what this estimate declares, and what it
    could not decide.

    The second list is the part that matters. A row contributing 0 because the
    architecture does not have that feature and a row contributing 0 because
    nobody recorded the dimension look identical in a total, and they mean
    opposite things: the first is a complete estimate of a simpler model, the
    second is an incomplete estimate of this one. `deepseek_v41`'s registry
    entry records neither `N_SHARED_EXP` nor `LEADING_DENSE_BLOCKS`, so its
    estimate is a FLOOR and says so, rather than quietly reading as a model
    without shared experts.
    """
    rows = []
    unknown = []

    def add(line, n=1):
        if n:
            rows.append((line, n))

    def optional(line, key, n):
        """A row whose presence depends on a dimension the registry may lack."""
        if key not in s:
            unknown.append(f"{key} is not in the registry, so `{line}` is uncounted")
            return
        if int(s.get(key) or 0) > 0:
            rows.append((line, n))

    moe = int(s.get("N_EXP") or 0) > 0
    kda = int(s.get("KDA_BLOCKS") or s.get("LINEAR_ATTN_HEADS") or 0) > 0
    # A LoRA-factored attention projection is the MLA signature, on EITHER side.
    # Testing only KV_LORA classified `deepseek_v41` as dense GQA -- it records
    # Q_LORA and HEAD_DIM but no KV_LORA -- and cost its estimate the five-way
    # MLA decomposition.
    mla = bool(s.get("KV_LORA") or s.get("Q_LORA"))
    rope = int(s.get("ROPE") or s.get("QK_ROPE_HEAD_DIM") or 0) > 0

    # --- model level ------------------------------------------------------
    add("embedding / get_rows at D")
    add("rmsnorm at D (attn_norm, ffn_norm and the final norm share it)")
    add("residual add at D")
    add("lm-head matvec at VOCAB")
    add("argmax over VOCAB")

    # --- attention --------------------------------------------------------
    if mla:
        # The standard MLA decomposition: a down-projection and an
        # up-projection on each of q and kv, plus the output projection.
        add("MLA projections: q_a, q_b, kv_a, kv_b, output", 5)
    else:
        add("attention projections: q, k, v, output", 4)
    add("rope", 1 if rope else 0)
    add("attention core (decode shape)")
    if kda:
        add("second attention core: the gated delta net")
        add("kda short causal convolution")
        # THE KDA PROJECTIONS. The rule counted the delta core and its convolution and
        # stopped, so the estimate was seven short: q, k and v to 8192 (64 heads x 128),
        # the two low-rank gate down-projections at 128, and their two up-projections
        # back to 8192. They are real tensors in the model's file and were emitted and
        # certified once the w8a8 arm stopped faulting above ~1280 outputs at k=4096.
        add("kda projections: q, k, v (to 8192), and the two low-rank gate pairs", 7)

    # --- the feed-forward path -------------------------------------------
    if moe:
        add("MoE router (scores, then top-k)")
        add("routed expert gate/up at MOE_INTER, and down at D", 2)
        optional("shared expert gate/up and down", "N_SHARED_EXP", 2)
        optional(
            "the leading dense blocks' gate/up at DENSE_FFN_INTER, and down at D",
            "LEADING_DENSE_BLOCKS",
            2,
        )
    else:
        add("dense FFN gate/up at DENSE_FFN_INTER, and down at D", 2)

    return rows, unknown


def counted(corpus):
    """What the corpus actually holds, per representation."""
    out = {}
    for sub, ext in (("tile", "tile"), ("pto", "pto"), ("ascendc", "cpp"), ("tile-rs", "rs")):
        d = BENCH / corpus / sub
        out[sub] = len(list(d.glob(f"*.{ext}"))) if d.is_dir() else 0
    return out


def sets():
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    out = {"playground16": PLAYGROUND}
    for corpus, key in (("deepseek_v41", "v41"), ("glm_5_3", "glm_5_3"), ("kimi_k3", "kimi_k3")):
        v = reg["sets"].get(key, {}).get("values")
        if v:
            out[corpus] = v
    return out


def main() -> int:
    rows = []
    for corpus, shapes in sets().items():
        count, floor, source = estimate(corpus, shapes)
        have = counted(corpus)
        check(corpus, count, have["tile"])
        inv, unknown = inventory(shapes)
        rows.append((corpus, count, floor, source, have, inv, unknown))

    w = max(len(c) for c, *_ in rows)
    print(f"{'corpus':<{w}}  kernels  tile   pto  ascendc  tile-rs")
    for corpus, count, floor, _, have, _, _ in rows:
        mark = "+" if floor else " "
        print(
            f"{corpus:<{w}}  {count:>6}{mark} {have['tile']:>4}  {have['pto']:>4}  "
            f"{have['ascendc']:>7}  {have['tile-rs']:>7}"
        )
    print("\n`+` is a LOWER BOUND, not the figure.")
    print(f"\nDerivation error, measured: {DERIVATION_ERROR}")
    print("\nWhat the inventory collapses -- the reason it is low:")
    for what, why in COLLAPSES:
        print(f"   * {what}")
        print(f"       {why}")

    for corpus, count, floor, source, have, inv, unknown in rows:
        print(f"\n--- {corpus}: {count}{'+' if floor else ''}")
        print(f"    source: {source}")
        if corpus in GROUND_TRUTH:
            derived = sum(n for _, n in inv)
            print(f"    the inventory below derives {derived}, which is NOT used here -- the known count is.")
        for line, n in inv:
            print(f"   {n:>2}  {line}")
        for u in unknown:
            print(f"    ?  {u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
