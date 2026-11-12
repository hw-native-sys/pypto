# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Emit the DeepSeek V4.1 Flash rows the corpus was missing.

The V4.1 corpus arrived as `ptobench` fixtures -- quantised GEMMs, `mx4pk`, a
`swiglu` -- which are real kernels but are not a projection of the inventory in
`registry.json`. Three of that inventory's rows had no kernel at all:

    embedding / get_rows at D
    residual add at D
    attention core (decode shape)

This generator adds exactly those, from the shape set the registry records
(`D`, `VOCAB`, `HEAD_DIM`), and nothing else. It refuses to run if the registry
stops carrying a shape set, for the same reason the GLM and K3 generators do:
an invented middle configuration measures nothing and looks exactly like a real
one afterwards.

# What it deliberately does not add

The inventory also names five MLA projections. Two of them are reachable from
sourced values here (`q_a` needs `Q_LORA`, `output` needs `D`), but the
up-projections need an output width the registry does not record -- the same
missing `KV_B_OUT` that blocks GLM 5.3 and Kimi K3. Adding them would mean
choosing a width, so they are left out and named here rather than guessed.

The routed-expert rows are also left alone: the corpus already holds their
quantised indexed forms (`gemm_i8_v41_*`, `mx4pk_v41_*`), which is the form the
engine actually runs, and an f32 corpus cannot hold the raw GGUF block buffer
that form takes.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "registry.json"
OUT = HERE.parent / "deepseek_v41" / "tile"

T = 1  # decode: one token, which is where the latency KPI lives


def c(sym, v, ty="i32"):
    return f"    %{sym} = llvm.mlir.constant({v} : {ty}) : {ty}\n"


def mod(name, nptr, body):
    args = ", ".join(f"%arg{i}: !llvm.ptr<1>" for i in range(nptr))
    return (
        "// DeepSeek V4.1 Flash (deepseek_v41), shapes read from the model's own GGUF.\n"
        "// Form: llvm.func + hacc.entry + __tile_* -- the shape the emitters consume.\n"
        f"module {{\n  llvm.func @{name}({args}) "
        f"attributes {{hacc.entry}} {{\n{body}\n    llvm.return\n  }}\n}}\n"
    )


def kernels(s):
    """{stem: mlir} for the three rows the corpus was missing."""
    D = s["D"]
    VOCAB = s["VOCAB"]
    HD = s["HEAD_DIM"]
    S = s["ATTN_S"]
    out = {}

    # --- residual add at the model width ------------------------------------
    b = (
        c("r", T)
        + c("cl", D)
        + c("z", 0)
        + """    %a = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @__tile_load_f32(%arg1, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_add_f32(%z, %a, %b, %r, %cl) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %o, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"residual_add_v41_{T}x{D}"] = mod("v41_residual_add", 3, b)

    # --- the embedding, VOCAB x D, indexed by a token id --------------------
    #
    # EXPRESSED AS A ROW COPY, not through `__tile_embedding_f32`, and the reason
    # is a contract rather than a bound. That intrinsic declares its weight table
    # as a TILE (`weight: u32`, a buf_id) and its doc line says it "gathers
    # `count` rows from (V, D) weight table by index", so the whole table has to
    # be resident. This model's table is 2647654400 bytes of f32 against a
    # 194560-byte unified buffer. No tiling closes a factor of 13608, and the
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
            f"gen_deepseek_v41: embedding at {T} rows. The row copy below reads ONE index and "
            "copies ONE row; more rows need a loop the .tile form cannot express "
            "without scf.for. Refusing rather than emitting a kernel that silently "
            "does the first row -- see deepseek_v41/attention_decode for what a "
            "degenerate shape costs."
        )
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
    out[f"embedding_v41_{VOCAB}x{D}"] = mod(f"v41_embedding_{D}", 3, b)

    # --- attention core at decode shape -------------------------------------
    #
    # REAL ATTENTION, not the degenerate stub this generator used to emit. Both
    # things it was missing are now sourced rather than invented:
    #
    #   * S is the engine's own attention block size -- 8 positions, the extent
    #     `ds4_attn_sink_batched` is launched at (bench/engine/tile/
    #     attn_sink_1head_1blk.tile: "ONE HEAD, 1 block(s) of 8, D=512").
    #   * The sink is a VIRTUAL POSITION, not a per-position bias. The engine's
    #     CPU reference pushes `attn_sinks[h]` as an extra score, scales every QK
    #     score by 1/sqrt(head_dim), and starts the weighted-sum loop at
    #     scores[1], so the sink joins the denominator and has no V
    #     (ds4-rs-ascend/crates/engine_core/src/attn_dispatch.rs:1428-1490).
    #     `__tile_attention_sink_f32` is the intrinsic for that contract; the PTO
    #     emitter's blocked form carries it and ascendc_emit has a 1-query arm.
    #
    # Q[1 x D], K[S x D], V[S x D]; K and V are the same buffer, the contract the
    # engine's one-head probe and the sink oracle both use. The PTO lowering
    # returns a [D x 1] col-major rowreduce tile, so it is stored at (D, 1): the
    # same contiguous D floats the oracle compares.
    b = (
        c("z", 0)
        + c("r", T)
        + c("s", S)
        + c("cl", HD)
        + """    %q = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %k = llvm.call @__tile_load_f32(%arg1, %s, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %v = llvm.call @__tile_load_f32(%arg1, %s, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_attention_sink_f32(%z, %q, %k, %v, %arg2, %s, %cl) : """
        + "(i32, i32, i32, i32, !llvm.ptr<1>, i32, i32) -> i32\n"
        + """    llvm.call @__tile_store_f32(%arg3, %o, %cl, %r) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"attention_decode_v41_{T}x{HD}"] = mod("v41_attn_decode", 4, b)

    # --- vocabulary argmax, per row, i32 indices ---------------------------
    #
    # TWO rows the corpus declares and the generator did not emit: the plain
    # vocab width and its next-power-of-two padded form. The old hand-captured
    # tiles loaded the whole 8 x 129280 plane (4.1 MB, 21x UB) and stored 8x8
    # f32 for 8 indices; `__tile_argmax_rows_i32` streams the plane per row and
    # writes the indices as i32. Ties resolve to the FIRST index, which is what
    # ORACLE 58 asserts with a per-row shifted tie.
    R = s["T"]  # 8 rows; T above is the decode batch and is 1
    for cols, stem in (
        (VOCAB, f"argmax_v41_{R}x{VOCAB}"),
        (131072, f"argmax_v41_{R}x131072_pad"),
    ):
        b = (
            c("r", R)
            + c("cl", cols)
            + f"""    llvm.call @__tile_argmax_rows_i32(%arg1, %arg0, %r, %cl) : """
            + "(!llvm.ptr<1>, !llvm.ptr<1>, i32, i32) -> ()"
        )
        out[stem] = mod(f"bm_argmax_{R}_{cols}", 2, b)

    return out


def main() -> int:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entry = reg["sets"]["v41"]
    shapes = entry.get("values")
    if not shapes or entry.get("status") != "sourced":
        print("v41 has no sourced shape set in the registry; refusing to invent any")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    written = kernels(shapes)
    for stem, text in written.items():
        (OUT / f"{stem}.tile").write_text(text, encoding="utf-8")
    print(f"wrote {len(written)} kernel(s) into {OUT}")
    for stem in written:
        print(f"  {stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
