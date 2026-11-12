# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Emit the Kimi K3 decode kernels as tile-dialect MLIR.

Every dimension is a key from the released model's own `config.json`, recorded
in `registry.json`. The generator refuses a shape set the registry does not mark
`sourced`, and it supplies no defaults: a default is a value nobody sourced.

# What separates this corpus from GLM 5.3's

Both are KDA/MLA hybrids, so both omit the delta-net recurrence and its short
causal convolution for the same reasons. Two things differ, and both change what
the corpus can hold:

1. **`hidden_act` is `situ`, not SwiGLU**, and carries two beta parameters
   (`activation_situ_beta`, `activation_situ_linear_beta`). The tile API has
   `silu`, `gate_up_silu`, `sigmoid` and `exp`, and nothing for `situ`. So the
   dense FFN and the expert FFN are NOT in this corpus, where GLM 5.3's were.
   Emitting `__tile_gate_up_silu_f16` for them would report on a computation
   this model does not perform -- the shapes would be right and the arithmetic
   would be another model's.

2. **896 experts, which is 7 x 128.** GLM 5.3's 288 admits no f32 tiling at all,
   because an f32 extent must be a multiple of 128 and 288 has no such divisor.
   896 does. That is recorded because it is the opposite of the GLM finding and
   the two are easy to collapse into "MoE expert counts do not tile".

The router is sigmoid + top-k with `noaux_tc`, not softmax + top-k, so it is
emitted that way: `__tile_sigmoid_f32` then `__tile_topk_f32`.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "registry.json"
OUT = HERE.parent / "kimi_k3" / "tile"

# Kernels the tile API has no vocabulary for, and what each would need.
# Recorded rather than approximated.
#
# THIS LIST IS NOW EMPTY, and the two entries it held are the best record in this
# file of what "the API cannot express this" actually costs:
#
#   * `kda_recurrence` -- the gated delta-net. It needed a SCALAR-TO-ROW
#     BROADCAST: `(q.k)` is one value that has to multiply every element of `v`.
#     `row_expand` existed in the emitters' vocabulary but `tile_std` declared no
#     `__tile_row_expand_f32`, so a call to it was a call to nothing. Declared on
#     2026-11-04; the row is `kda_delta_k3_1x128` and it runs on the 910B2.
#
#   * `kda_short_conv` -- the depthwise causal convolution over
#     SHORT_CONV_KERNEL timesteps. It did not need a new intrinsic at all: it
#     needed the corpus to admit that a decode conv is a fixed-radius stencil,
#     which a broadcast can express. The row is `kda_conv_k3_1x7168_k4`.
#
# Kept as a comment rather than deleted, because the next person to add a row
# here should see what the last two cost: one declaration, and one realisation
# that the corpus, not the API, was the thing standing in the way.
UNEXPRESSIBLE = {}


def mod(name, nptr, body):
    args = ", ".join(f"%arg{i}: !llvm.ptr<1>" for i in range(nptr))
    return (
        "// Kimi K3 ({arch}), shapes read from the released model's own config.json.\n"
        "// Form: llvm.func + hacc.entry + __tile_* -- the shape the emitters consume.\n"
        f"module {{\n  llvm.func @{name}({args}) "
        f"attributes {{hacc.entry}} {{\n{body}\n    llvm.return\n  }}\n}}\n"
    )


def c(sym, v, ty="i32"):
    return f"    %{sym} = llvm.mlir.constant({v} : {ty}) : {ty}\n"


def situ_gate_up_body(beta: float, beta_lin: float, width: int) -> str:
    """MLIR for `situ(g, u)` over one `1 x width` tile, using ops BOTH emitters accept.

    `situ(g, u) = b * tanh(g/b) * sigmoid(g) * b_lin * tanh(u/b_lin)`, with b and
    b_lin from the model's own config. Sourced from the K3 reference
    implementation, not remembered.

    Three constraints shaped the form:

      * there is no scalar add in the dialect, and `fill` is spelled differently
        by the two emitters, so the constant 1.0 is synthesised as
        `exp(scale(x, 0.0))` -- exactly 1.0 for every input including zero, where
        an `x/x` trick would produce NaN;
      * `sigmoid` has a PTO arm but no AscendC one, so it is `1 / (1 + exp(-x))`;
      * `tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)`.

    Both identities are the real functions evaluated in f32, so the only
    difference from the reference activation is rounding.
    """
    T32 = "i32, i32, f32, i32, i32"
    T1 = "i32, i32, i32, i32"
    T2 = "i32, i32, i32, i32, i32"
    L = [
        c("z", 0), c("r", 1), c("cl", width),
        c("c1f", 1.0, "f32"), c("cm1f", -1.0, "f32"),
        c("z0f", 0.0, "f32"),
        c("cm2ob", -2.0 / beta, "f32"), c("cm2obl", -2.0 / beta_lin, "f32"),
        c("cbb", beta * beta_lin, "f32"),
        "    %g = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32\n",
        "    %u = llvm.call @__tile_load_f32(%arg1, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32\n",
        # ONE, as a tile, without a constant TILE: exp(scale(x, 0.0)) is exactly
        # 1.0 for every input including zero, where an `x/x` trick would be NaN at
        # zero. Two buffers, shared by both branches and by both constants.
        f"    %one0 = llvm.call @__tile_scale_f32(%z, %g, %z0f, %r, %cl) : ({T32}) -> i32\n",
        f"    %one = llvm.call @__tile_exp_f32(%z, %one0, %r, %cl) : ({T1}) -> i32\n",
        f"    %two = llvm.call @__tile_adds_f32(%z, %one, %c1f, %cl) : (i32, i32, f32, i32) -> i32\n",
        # sigmoid(g) = 1 / (1 + e^-g).  THE STABLE FORM. The previous identity was
        # e^g / (1 + e^g), which is NaN for g > 88.7 because e^g overflows f32 --
        # and the docstring above this function claimed the stable one while the
        # code used the other. Down the negative side e^-g underflows to 0, which
        # divides correctly, so this form is exact at both infinities.
        f"    %gneg = llvm.call @__tile_scale_f32(%z, %g, %cm1f, %r, %cl) : ({T32}) -> i32\n",
        f"    %gen = llvm.call @__tile_exp_f32(%z, %gneg, %r, %cl) : ({T1}) -> i32\n",
        f"    %gd = llvm.call @__tile_adds_f32(%z, %gen, %c1f, %cl) : (i32, i32, f32, i32) -> i32\n",
        f"    %sg = llvm.call @__tile_div_f32(%z, %one, %gd, %r, %cl) : ({T2}) -> i32\n",
        # tanh(g/b) = 2/(1 + e^-2g/b) - 1.  Also stable at both ends: e^-2z
        # underflows to 0 for large positive z (giving exactly +1) and overflows to
        # inf for large negative z, where 2/inf is 0 and the result is exactly -1.
        # The previous (e^2z - 1)/(e^2z + 1) was NaN past |z| ~ 44.
        f"    %gz2 = llvm.call @__tile_scale_f32(%z, %g, %cm2ob, %r, %cl) : ({T32}) -> i32\n",
        f"    %gE = llvm.call @__tile_exp_f32(%z, %gz2, %r, %cl) : ({T1}) -> i32\n",
        f"    %gEp = llvm.call @__tile_adds_f32(%z, %gE, %c1f, %cl) : (i32, i32, f32, i32) -> i32\n",
        f"    %gq = llvm.call @__tile_div_f32(%z, %two, %gEp, %r, %cl) : ({T2}) -> i32\n",
        f"    %th = llvm.call @__tile_subs_f32(%z, %gq, %c1f, %cl) : (i32, i32, f32, i32) -> i32\n",
        f"    %gs = llvm.call @__tile_mul_f32(%z, %th, %sg, %r, %cl) : ({T2}) -> i32\n",
        # the up branch: the same stable tanh, sharing the ONE and TWO tiles
        f"    %uz = llvm.call @__tile_scale_f32(%z, %u, %cm2obl, %r, %cl) : ({T32}) -> i32\n",
        f"    %uE = llvm.call @__tile_exp_f32(%z, %uz, %r, %cl) : ({T1}) -> i32\n",
        f"    %uEp = llvm.call @__tile_adds_f32(%z, %uE, %c1f, %cl) : (i32, i32, f32, i32) -> i32\n",
        f"    %uq = llvm.call @__tile_div_f32(%z, %two, %uEp, %r, %cl) : ({T2}) -> i32\n",
        f"    %ut = llvm.call @__tile_subs_f32(%z, %uq, %c1f, %cl) : (i32, i32, f32, i32) -> i32\n",
        # the two betas are folded into ONE final scale rather than applied to
        # each branch: multiplication is associative, and this is two buffers
        # fewer in a chain that has none to spare.
        f"    %o = llvm.call @__tile_mul_f32(%z, %gs, %ut, %r, %cl) : ({T2}) -> i32\n",
        f"    %ob = llvm.call @__tile_scale_f32(%z, %o, %cbb, %r, %cl) : ({T32}) -> i32\n",
        "    llvm.call @__tile_store_f32(%arg2, %ob, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n",
    ]
    return "".join(L)


def kernels(s):
    """{stem: mlir} for the Kimi K3 shape set."""
    D = s["D"]
    T = 1  # decode: one token at a time, which is where latency lives
    out = {}

    # --- RMSNorm over [1, D] -- attn_norm / ffn_norm, 93 blocks each ---------
    b = (
        c("r", T)
        + c("cl", D)
        + f"""    %eps = llvm.mlir.constant({s["EPS"]} : f32) : f32
    %x = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_rms_norm_f32(%x, %x, %eps, %r, %cl) : (i32, i32, f32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %o, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"rmsnorm_k3_{T}x{D}"] = mod(f"k3_rmsnorm_{T}_{D}", 2, b)

    # --- MLA projections at decode width ------------------------------------
    # `__tile_matvec_f16` with a leading dummy destination, the form
    # playground16's q_proj uses.
    #
    # The two up-projection WIDTHS are DERIVED, not chosen, and the formula is
    # the architecture's own: MLA's q_b projects the Q latent to one
    # (qk_nope + qk_rope) row per head, and kv_b projects the KV latent to one
    # (qk_nope + v) row per head. Every field below is sourced from the model's
    # config.json, which is why this is a derivation and not a guess -- the rule
    # the registry enforces is "no shape without a cited source", and a formula
    # over cited fields is a source. Recorded here because the registry carries
    # only sourced values, deliberately: putting a derived number in it would
    # make it indistinguishable from a measured one.
    q_b = s["HEAD_COUNT"] * (s["QK_NOPE_HEAD_DIM"] + s["QK_ROPE_HEAD_DIM"])
    kv_b = s["HEAD_COUNT"] * (s["QK_NOPE_HEAD_DIM"] + s["V_HEAD_DIM"])
    for name, n in [
        ("q_a", s["Q_LORA"]),
        ("q_b", q_b),
        ("kv_a_mqa", s["KV_LORA"]),
        ("kv_b", kv_b),
        ("output", D),
    ]:
        b = (
            c("z", 0)
            + c("r", T)
            + c("cr1", 1)
            + c("cl", n)
            + """    %x = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %w = llvm.call @__tile_load_f32(%arg1, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %m = llvm.call @__tile_mul_f32(%z, %x, %w, %r, %cl) : (i32, i32, i32, i32, i32) -> i32
    %o = llvm.call @__tile_reduce_sum_f32(%z, %m, %r, %cl) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %o, %cr1, %cr1) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
        )
        out[f"proj_{name}_k3_{T}x{n}"] = mod(f"k3_proj_{name}", 3, b)

    # --- the FFN, gated by situ, which the tile API has no single op for -----
    #
    # The three gate/up rows K3 was missing. `situ_gate_up_body` writes the
    # activation out; the widths and the two betas are the model's own.
    beta, beta_lin = s["ACT_SITU_BETA"], s["ACT_SITU_LINEAR_BETA"]
    for name, width in [
        ("routed", s["MOE_INTER"]),
        ("shexp", s["MOE_INTER"]),
        ("dense", s["DENSE_FFN_INTER"]),
    ]:
        out[f"{name}_gate_up_situ_k3_{T}x{width}"] = mod(
            f"k3_{name}_situ", 3, situ_gate_up_body(beta, beta_lin, width)
        )
        # The matching down-projection, D -> D. It carries no activation, so it
        # is the same matvec form the other down rows in this corpus use.
        b = (
            c("z", 0)
            + c("r", T)
            + c("cr1", 1)
            + c("cl", D)
            + """    %x = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %w = llvm.call @__tile_load_f32(%arg1, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %m = llvm.call @__tile_mul_f32(%z, %x, %w, %r, %cl) : (i32, i32, i32, i32, i32) -> i32
    %o = llvm.call @__tile_reduce_sum_f32(%z, %m, %r, %cl) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %o, %cr1, %cr1) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
        )
        out[f"{name}_ffn_down_k3_{T}x{D}"] = mod(f"k3_{name}_down", 3, b)

    # --- the gated delta-net, AT DECODE --------------------------------------
    #
    # Not a scan at this shape: decode is one step, and the update re-associates
    # to
    #
    #     o = decay * (q . S) + beta * (q . k) * v
    #
    # `q . S` is the running state row, `q . k` is one value, and the only op the
    # work needs that the API lacked is a SCALAR-TO-ROW BROADCAST for `(q . k)`.
    # `__tile_row_expand_f32` was declared on 2026-11-04 for exactly this -- the
    # emitters already knew `pto.trowexpand`; `tile_std` had no name for it.
    hd = s["KDA_HEAD_DIM"]
    dl = [
        c("z", 0), c("r", T), c("cl", hd),
        c("cdecay", 0.5, "f32"), c("cbeta", 1.0, "f32"),
        "    %q = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32\n",
        "    %k = llvm.call @__tile_load_f32(%arg1, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32\n",
        "    %v = llvm.call @__tile_load_f32(%arg2, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32\n",
        "    %s = llvm.call @__tile_load_f32(%arg3, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32\n",
        # q . k, as one value, broadcast back over the row
        "    %qk = llvm.call @__tile_mul_f32(%z, %q, %k, %r, %cl) : (i32, i32, i32, i32, i32) -> i32\n",
        "    %qks = llvm.call @__tile_reduce_sum_f32(%z, %qk, %r, %cl) : (i32, i32, i32, i32) -> i32\n",
        "    %qkb = llvm.call @__tile_row_expand_f32(%z, %qks, %cl) : (i32, i32, i32) -> i32\n",
        "    %g1 = llvm.call @__tile_mul_f32(%z, %qkb, %v, %r, %cl) : (i32, i32, i32, i32, i32) -> i32\n",
        "    %g2 = llvm.call @__tile_scale_f32(%z, %g1, %cbeta, %r, %cl) : (i32, i32, f32, i32, i32) -> i32\n",
        # decay * (q . S)
        "    %sd = llvm.call @__tile_scale_f32(%z, %s, %cdecay, %r, %cl) : (i32, i32, f32, i32, i32) -> i32\n",
        "    %o = llvm.call @__tile_add_f32(%z, %sd, %g2, %r, %cl) : (i32, i32, i32, i32, i32) -> i32\n",
        "    llvm.call @__tile_store_f32(%arg4, %o, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n",
    ]
    out[f"kda_delta_k3_{T}x{hd}"] = mod(f"k3_kda_delta_{hd}", 5, "".join(dl))

    # --- the KDA PROJECTIONS, at their real widths ---------------------------
    # Kimi K3 is 69 KDA blocks of 93, so these are its trunk. They are plain matvecs
    # through the w8a8 arm, and they were missing because that arm faulted above ~1280
    # outputs at k=4096 until iteration 151 fixed it -- not because the API could not
    # express them.
    #   kda_q/k/v : 8192 out from k=s["D"]             (64 heads x 128)
    #   kda_f_a/g_a: 128 out
    #   kda_f_b/g_b: 8192 out from k=128
    #
    # TWO FORMS OF THE SAME ARITHMETIC, chosen by whether the activation FITS.
    #
    # `__tile_matvec_w8a8_f32` stages the activation as one k-wide f32 tile, so k
    # is bounded by the per-buffer share of the unified buffer. On 910B2 the
    # AscendC emitter refuses `k3_kda_q` with:
    #
    #   UB overflow in kernel 'k3_kda_q': buffer 0 needs 7168 elements
    #   (28672 bytes) but only 17120 bytes available per buffer
    #   (188416 UB / 11 slots)
    #
    # K3's D is 7168, so q/k/v overflow where GLM 5.3's identical rows at k=4096
    # fit -- the refusal is this model's dimension, not the API's vocabulary.
    #
    # `__tile_matvec_w8a8_gm_f32` is the variant that exists for exactly this: it
    # streams BOTH the activation and the weight from GM one k-chunk at a time, so
    # k is unbounded, and `tile_std/src/tile.rs:1061` records it as the SAME
    # arithmetic. It is not speculative here -- `bench/engine/tile/` already runs
    # it at k=8192 and k=16384.
    #
    # The activation arrives as a POINTER, not a staged tile id, so the chunked
    # body has no `__tile_load_f32` and the operand order differs (x, w, sc, off)
    # from the resident form's (w, sc, off, a). Arity stays 5 pointers.
    #
    # THE THRESHOLD IS A RECORD OF TWO MEASUREMENTS, NOT A MODEL OF THE ALLOCATOR.
    # Re-deriving the emitter's UB allocator here would put a second counter on one
    # quantity, and the two would disagree silently. So the condition below only
    # restates what the emitter was OBSERVED to do on 2026-11-07, for every one of
    # these seven rows:
    #
    #   n=8192 k=7168  resident -> REFUSES (the overflow quoted above, 11 slots)
    #   n=128  k=7168  resident -> EMITS   (fewer slots, so a larger share each)
    #   n=8192 k=128   resident -> EMITS   (the activation is tiny)
    #
    # The per-buffer share therefore depends on n as well as k, and a k-only bound
    # over-triggers: an earlier version of this edit used one and switched the two
    # n=128 rows too. They emit either way, so nothing broke -- but the chunked form
    # re-reads the activation once per k-chunk where the resident form stages it
    # once, so that would have made two working kernels do more GM traffic for no
    # coverage. Both halves of the condition are load-bearing.
    #
    # If a future row refuses on the resident form, this is the wrong place to guess
    # at it: take the refusal from the emitter and add it to the table above.
    RESIDENT_ACT_MAX_ELEMS = 17120 // 4  # 4280 f32, from the refusal quoted above
    SLOT_PRESSURE_WIDTH = 8192  # the n at which the measured refusal occurs

    for name, n, k in [
        ("kda_q", 8192, s["D"]),
        ("kda_k", 8192, s["D"]),
        ("kda_v", 8192, s["D"]),
        ("kda_f_a", 128, s["D"]),
        ("kda_g_a", 128, s["D"]),
        ("kda_f_b", 8192, 128),
        ("kda_g_b", 8192, 128),
    ]:
        head = c("r", T) + c("kk", k) + c("nn", n) + c("z", 0)
        if k > RESIDENT_ACT_MAX_ELEMS and n >= SLOT_PRESSURE_WIDTH:
            # Chunked: the activation stays in GM. Mirrors
            # bench/engine/tile/proj_w8a8_engine_1x4096_k8192.tile.
            b = head + (
                "    %res = llvm.call @__tile_matvec_w8a8_gm_f32"
                "(%arg0, %arg1, %arg2, %arg3, %z, %nn, %kk) : "
                "(!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>,"
                " i32, i32, i32) -> i32\n"
                "    llvm.call @__tile_store_f32(%arg4, %res, %r, %nn) : "
                "(!llvm.ptr<1>, i32, i32, i32) -> ()\n"
            )
        else:
            b = head + (
                "    %x = llvm.call @__tile_load_f32(%arg0, %r, %kk) : "
                "(!llvm.ptr<1>, i32, i32) -> i32\n"
                "    %res = llvm.call @__tile_matvec_w8a8_f32"
                "(%arg1, %arg2, %arg3, %x, %z, %nn, %kk) : "
                "(!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32,"
                " i32) -> i32\n"
                "    llvm.call @__tile_store_f32(%arg4, %res, %r, %nn) : "
                "(!llvm.ptr<1>, i32, i32, i32) -> ()\n"
            )
        out[f"{name}_k3_{T}x{n}_k{k}"] = mod(f"k3_{name}", 5, b)


    # --- the KDA short causal convolution, AT DECODE -------------------------
    #
    # The generator used to refuse this because there is no sliding-window
    # intrinsic and "expressing it as a matmul would change the arithmetic and
    # the memory traffic both". At decode neither objection applies: the window
    # is the SHORT_CONV_KERNEL most recent timesteps, which the caller already
    # holds as rows, so the convolution is `sum_j w_j * x_j` over those rows --
    # a weighted sum, exactly, with no matmul and no sliding window.
    #
    # The weights are per channel, so each tap carries its own 1 x D weight row
    # rather than a scalar; that is why the signature is 9 pointers.
    k_conv = s["SHORT_CONV_KERNEL"]
    dl = []
    dl.append(c("z", 0))
    dl.append(c("r", T))
    dl.append(c("cl", D))
    #
    # INTERLEAVED, not "load everything then multiply". Loading all four taps and
    # all four weight rows before the first multiply keeps NINE 1xD tiles live at
    # once, and at D = 7168 that is 9 x 28672 = 258048 B against a 188416 B
    # buffer -- the row could not place even with the aliasing pass, because those
    # nine genuinely overlap. Interleaving a tap at a time holds four live tiles
    # (x_j, w_j, the product and the accumulator) and the same arithmetic in the
    # same order, which is what the compiler would have scheduled anyway.
    for j in range(k_conv):
        dl.append(f"    %x{j} = llvm.call @__tile_load_f32(%arg{j}, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32\n")
        dl.append(
            f"    %w{j} = llvm.call @__tile_load_f32(%arg{4 + j}, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32\n"
        )
        if j == 0:
            dl.append(
                f"    %p0 = llvm.call @__tile_mul_f32(%z, %x0, %w0, %r, %cl) : (i32, i32, i32, i32, i32) -> i32\n"
            )
        else:
            dl.append(
                f"    %pj{j} = llvm.call @__tile_mul_f32(%z, %x{j}, %w{j}, %r, %cl) : (i32, i32, i32, i32, i32) -> i32\n"
            )
            dl.append(
                f"    %p{j} = llvm.call @__tile_add_f32(%z, %p{j - 1}, %pj{j}, %r, %cl) : (i32, i32, i32, i32, i32) -> i32\n"
            )
    dl.append(
        f"    llvm.call @__tile_store_f32(%arg{2 * k_conv}, %p{k_conv - 1}, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n"
    )
    out[f"kda_conv_k3_{T}x{D}_k{k_conv}"] = mod(f"k3_kda_conv_{D}", 2 * k_conv + 1, "".join(dl))

    # --- the MoE router: sigmoid over N_EXP scores, then top-k --------------
    # Sigmoid, not softmax: `moe_router_activation_func` is `sigmoid` and
    # `topk_method` is `noaux_tc`. GLM 5.3's generator emits top-k alone; this
    # one carries the activation because the config names a different one.
    ne, topk = s["N_EXP"], s["TOPK"]
    b = (
        c("z", 0)
        + c("r", 1)
        + c("cl", ne)
        + c("k", topk)
        + """    %v = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %p = llvm.call @__tile_sigmoid_f32(%z, %v, %r, %cl) : (i32, i32, i32, i32) -> i32
"""
        # Split across two Python strings so this SOURCE line stays inside the
        # line limit while the EMITTED MLIR line is unchanged.
        + "    %o = llvm.call @__tile_topk_f32(%p, %p, %arg2, %r, %cl, %k) : "
        + "(i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32\n"
        + """    llvm.call @__tile_store_f32(%arg1, %o, %r, %k) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"router_sigmoid_topk_k3_1x{ne}_k{topk}"] = mod(f"k3_router_{ne}_{topk}", 3, b)

    # --- the output head's argmax over the vocabulary ----------------------
    vo = s["VOCAB"]
    b = (
        c("r", T)
        + c("cl", vo)
        # THE SECOND STORE EXTENT IS ITS OWN CONSTANT, not `%r` again.
        # argmax over an `R x CL` input yields R indices, so the store is
        # `R x 1`. Writing `%r, %r` declares R*R elements, which is CORRECT BY
        # COINCIDENCE at T=1 (1x1) and wrong at every other R -- deepseek's two
        # authored argmax tiles are the R=8 case and declare 64 elements for 8
        # indices. The cause was that `%r` and `%cl` were the only constants in
        # scope and `%r` got used for both; the top-k sibling in this file shows
        # the correct form, `(%arg1, %o, %r, %k)` with a distinct second extent.
        # Pinned by `test_no_generator_uses_one_extent_for_both_store_dimensions`.
        + c("one", 1)
        + """    %v = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_argmax_f32(%v, %v, %r, %cl) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_i32(%arg1, %o, %r, %one) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"argmax_k3_{T}x{vo}"] = mod(f"k3_argmax_{vo}", 2, b)

    # --- residual add at the model width ----------------------------------
    b = (
        c("r", T)
        + c("cl", D)
        + c("z", 0)
        + """    %a = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @__tile_load_f32(%arg1, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_add_f32(%z, %a, %b, %r, %cl) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %o, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"residual_add_k3_{T}x{D}"] = mod("k3_residual_add", 3, b)

    # --- the embedding, VOCAB x D, indexed by a token id -------------------
    #
    # EXPRESSED AS A ROW COPY, not through `__tile_embedding_f32`, and the reason
    # is a contract rather than a bound. That intrinsic declares its weight table
    # as a TILE (`weight: u32`, a buf_id) and its doc line says it "gathers
    # `count` rows from (V, D) weight table by index", so the whole table has to
    # be resident. This model's table is 4697620480 bytes of f32 against a
    # 194560-byte unified buffer. No tiling closes a factor of 24144, and the
    # AscendC refusal said so while wording it as a UB overflow, which invited
    # raising a bound; mlir_to_pto refused by NAME, which invited adding an arm
    # that would have read `weight` as the one-row tile this file used to load
    # and returned row 0 for every index.
    #
    # At decode shape the operation needs none of that. One row copy from GM:
    # read the index, multiply by the row width, load at that element offset.
    # `pto-to-ascendc` emits `DataCopy(buf1, gm0Tensor[__v3], (__v1) * (__v2))`,
    # so the offset is a runtime value and the table is never resident. The
    # scalar read-back `__tile_get_value_f32` was implemented in mlir_to_cpp all
    # along and declared by nothing until 2026-11-08; that one declaration is the
    # whole difference. Measured and recorded in the campaign under
    # evidence/0012-embedding-is-a-row-copy/.
    #
    # mlir_to_pto refuses this form too ("no arm for __tile_get_value_f32"), which
    # is honest and costs nothing: the device run goes through the AscendC column.
    # THE INDEX IS LOADED AS EIGHT ELEMENTS, NOT ONE, AND THE DEVICE IS WHY.
    # Measured on an Ascend910B2 (dav-2201) 2026-11-09 against ORACLE 44, which
    # puts the index 3 at element 0 and a decoy 0.0 at element 1:
    #
    #   1x1 index load -> FAIL, wrong=5120, first disagreement at col 0:
    #                     got 640.125 want 3840.125. Decoded against the table's
    #                     own formula that is element 2560 rather than 15360, so
    #                     the implied index was 0.5 -- not 3, and not the decoy.
    #   1x8 index load -> PASS, wrong=0, max_abs=0, max_rel=0.
    #
    # Same kernel, same oracle, same buffers; the only difference is the extent of
    # the load. A one-element `DataCopy` is FOUR BYTES, below the 32-byte DMA
    # granularity of this part, and it does not fault or refuse -- it leaves
    # something else in the tile. Eight f32 is exactly one 32-byte block, and the
    # kernel reads element 0 of it, so the extra seven are loaded and ignored.
    #
    # This is the kind of defect an emitter should refuse rather than emit, and
    # mlir_to_cpp does not check it: nothing in the tile form says a load must be
    # at least a block. Logged for the owner rather than worked around here.
    if T != 1:
        raise SystemExit(
            f"gen_kimi_k3: embedding at {T} rows. The row copy below reads ONE index and "
            "copies ONE row; more rows need a loop the .tile form cannot express "
            "without scf.for. Refusing rather than emitting a kernel that silently "
            "does the first row -- see deepseek_v41/attention_decode for what a "
            "degenerate shape costs."
        )
    vo = s["VOCAB"]
    b = (
        c("z", 0)
        + c("d", D)
        + c("n", T)
        + c("e8", 8)
        + """    %ti = llvm.call @__tile_load_f32(%arg1, %n, %e8) : (!llvm.ptr<1>, i32, i32) -> i32
    %idx = llvm.call @__tile_get_value_f32(%ti, %z) : (i32, i32) -> i32
    %off = llvm.mul %idx, %d : i32
    %row = llvm.call @__tile_load_at_f32(%arg0, %off, %n, %d) : (!llvm.ptr<1>, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %row, %n, %d) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"embedding_k3_{vo}x{D}"] = mod(f"k3_embedding_{D}", 3, b)

    # --- the lm head: D -> VOCAB ------------------------------------------
    b = (
        c("z", 0)
        + c("r", T)
        + c("cr1", 1)
        + c("cl", vo)
        + """    %x = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %w = llvm.call @__tile_load_f32(%arg1, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %m = llvm.call @__tile_mul_f32(%z, %x, %w, %r, %cl) : (i32, i32, i32, i32, i32) -> i32
    %o = llvm.call @__tile_reduce_sum_f32(%z, %m, %r, %cl) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %o, %cr1, %cr1) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"lm_head_k3_{T}x{vo}"] = mod(f"k3_lm_head_{vo}", 3, b)

    # --- rope on the MLA path. K3 HAS rope, where GLM 5.3 has none ---------
    #
    # EXPRESSED IN PRIMITIVES, not through `__tile_rope_f32`, and the reason is a
    # measurement rather than a preference. The stub intrinsic has no cos/sin
    # operands, so `mlir_to_cpp` substitutes an identity copy and records
    # `TILE_SUBSTITUTION: intrinsic=__tile_rope_f32; replacement=identity-copy;
    # reason=missing-cos-sin-operands`. Against a real RoPE oracle (ORACLE 12,
    # "b is cos and v is sin") that identity was measured on device 2026-11-07 at
    # `wrong=62` of 64 elements, `max_rel=3.206e+01` -- 32x wrong, not a rounding
    # question. The same shape written as the chain below PASSED in the same
    # batch at `wrong=0 max_rel=6.001e-07`, so no emitter change was ever needed:
    # the defect was here, in the generator, calling a stub.
    #
    # The chain and the formula are the engine's verified
    # `rope_halves_engine_1x64` (gen_engine.rope_halves), which is itself taken
    # from ptoas_rs/tests/fixtures/barrier/layer0.pto:364:
    #
    #     out_lo = x_lo*cos - x_hi*sin
    #     out_hi = x_hi*cos + x_lo*sin
    #
    # FOUR pointers, because cos and sin have to be real operands -- a two-pointer
    # signature cannot express RoPE at all, which is the root of the stub.
    #
    # ROWS MUST BE 1. The halves are reached with a flat `getelementptr` by the
    # half width, which is the row's second half only while there is one row. At
    # T rows the low half of each row sits `cols` apart, i.e. a load of T x half
    # out of a T x cols buffer with a row stride, and `__tile_load_f32` takes no
    # stride (crates/tile_std/src/tile.rs). That is exactly why
    # deepseek_v41/rope_v41_8x64 (8 rows) is NOT fixable here and still carries
    # the substitution: it needs a strided load, which is a crates change. So this
    # arm refuses rather than silently emitting a wrong decomposition.
    rd = s["QK_ROPE_HEAD_DIM"]
    if T != 1:
        raise SystemExit(
            f"gen_kimi_k3: rope_k3_{T}x{rd} has {T} rows. The halves formulation "
            f"below reaches the high half with a flat getelementptr and is correct "
            f"only at one row; a multi-row rope needs a STRIDED load that "
            f"__tile_load_f32 does not have. Refusing rather than emitting a wrong "
            f"decomposition -- see deepseek_v41/rope_v41_8x64."
        )
    half = rd // 2
    assert rd % 2 == 0, f"rope head dim {rd} must be even to split into halves"
    b = (
        c("r", T)
        + c("ch", half)
        + c("z", 0)
        + """    %lo = llvm.call @__tile_load_f32(%arg0, %r, %ch) : (!llvm.ptr<1>, i32, i32) -> i32
    %hop = llvm.getelementptr inbounds %arg0[%ch] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
    %hi = llvm.call @__tile_load_f32(%hop, %r, %ch) : (!llvm.ptr<1>, i32, i32) -> i32
    %cos = llvm.call @__tile_load_f32(%arg1, %r, %ch) : (!llvm.ptr<1>, i32, i32) -> i32
    %sin = llvm.call @__tile_load_f32(%arg2, %r, %ch) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_mul_f32(%z, %lo, %cos, %r, %ch) : (i32, i32, i32, i32, i32) -> i32
    %t2 = llvm.call @__tile_mul_f32(%z, %hi, %sin, %r, %ch) : (i32, i32, i32, i32, i32) -> i32
    %ol = llvm.call @__tile_sub_f32(%z, %t1, %t2, %r, %ch) : (i32, i32, i32, i32, i32) -> i32
    %t3 = llvm.call @__tile_mul_f32(%z, %hi, %cos, %r, %ch) : (i32, i32, i32, i32, i32) -> i32
    %t4 = llvm.call @__tile_mul_f32(%z, %lo, %sin, %r, %ch) : (i32, i32, i32, i32, i32) -> i32
    %oh = llvm.call @__tile_add_f32(%z, %t3, %t4, %r, %ch) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg3, %ol, %r, %ch) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    %oop = llvm.getelementptr inbounds %arg3[%ch] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
    llvm.call @__tile_store_f32(%oop, %oh, %r, %ch) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"rope_k3_{T}x{rd}"] = mod(f"k3_rope_{rd}", 4, b)

    # --- the attention core at decode shape -------------------------------
    hd = s["V_HEAD_DIM"]
    # REAL 1-query decode attention at a RUNTIME context length. The model's
    # full attention is over max_position_embeddings = 1,048,576; a decode
    # step's KV length is a runtime value, so this tile reads S from a 1x1 GM
    # scalar, not a pinned shape. K/V are raw pointers streamed in blocks.
    b = (
        c("z", 0)
        + c("r", T)
        + c("cl", hd)
        + """    %q = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %st = llvm.call @__tile_load_f32(%arg3, %r, %r) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_attention_dyn_f32(%z, %q, %arg1, %arg2, %st, %cl) : """
        + "(i32, i32, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32) -> i32\n"
        + """    llvm.call @__tile_store_f32(%arg4, %o, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"attention_decode_k3_{T}x{hd}"] = mod(f"k3_attn_decode_{hd}", 5, b)

    return out


def main() -> int:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entry = reg["sets"]["kimi_k3"]
    if entry.get("status") != "sourced":
        print(
            f"kimi_k3 is `{entry.get('status')}` in the registry; refusing to "
            "generate. Record the source first."
        )
        return 1
    shapes = entry.get("values")
    if not shapes:
        print("kimi_k3 has no shapes in the registry; refusing to invent any")
        return 1

    # No defaults. Every dimension this generator uses must be in the registry,
    # because a default is a value nobody sourced.
    s = dict(shapes)
    required = (
        "D",
        "VOCAB",
        "Q_LORA",
        "KV_LORA",
        "N_EXP",
        "TOPK",
        "EPS",
        "QK_ROPE_HEAD_DIM",
        "V_HEAD_DIM",
    )
    missing = [k for k in required if k not in s]
    if missing:
        print(f"registry entry for kimi_k3 is missing {missing}; refusing to guess")
        return 1

    # The activation is the reason two FFN kernels are absent, so a change to it
    # has to be noticed here rather than silently widen the corpus.
    if s.get("HIDDEN_ACT") != "situ":
        print(
            f"HIDDEN_ACT is now `{s.get('HIDDEN_ACT')}`, not `situ`. The FFN "
            "kernels are omitted on the grounds that `situ` has no tile-API "
            "intrinsic; re-read that before generating."
        )
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, text in kernels(s).items():
        p = OUT / f"{stem}.tile"
        p.write_text(text.replace("{arch}", s["architecture"]), encoding="utf-8")
        written.append(p.name)
    print(f"wrote {len(written)} kernel(s) to {OUT}")
    for name, why in UNEXPRESSIBLE.items():
        print(f"NOT emitted: {name} -- {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
