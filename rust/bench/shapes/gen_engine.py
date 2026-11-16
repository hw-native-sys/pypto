# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernels the Rust SERVING ENGINE launches, at the shapes it launches them.

Why this corpus exists. The engine's NPU kernels come from one hand-written
`layer0.cpp` ELF (`engine_ascend::device::LAYER0_KERNELS`). The goal is to run
the three models on kernels the Rust emitter produced instead, so each engine
kernel needs an artefact with ITS shape -- and several shapes appear nowhere in
the model corpora, because a model row is named after the block it came from
rather than after the engine slot it fills.

This generator covers only the engine's own slots. A row here is not a model
row: it exists because `rms_norm_512` is launched, not because GLM-5.3 has a
512-wide normalisation.

Nothing beyond the shapes is invented: the body is the same `__tile_rms_norm_f32`
chain every model corpus uses for its normalisation.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "engine" / "tile"
MANIFEST = HERE.parent / "engine" / "manifest.json"

# The engine's rms_norm slots with no model-corpus counterpart. `eps` is the
# engine's own (1e-6, `ds4_rmsmul_*`), not a model's.
SLOTS = [
    ("rms_norm_512", 512),
    ("rms_norm_1024", 1024),
    # `rms_norm_4096` was credited by `rmsnorm_glm53_1x4096`, which is ALSO a pure rms_norm. The
    # engine's socket is `ds4_rmsmul_4096(x, gamma, out)`, so the slot needs its own row here.
    ("rms_norm_4096", 4096),
    # `hc_norm` is this chain at 1x16384 -- read off the only copy of that kernel in
    # this repository, ptoas_rs/tests/fixtures/barrier/layer0.pto:1157, whose body is
    # tload -> tmul -> trowsum -> tmuls(1/16384) -> tadds(1e-6) -> tsqrt -> trecip ->
    # trowexpand -> tmul -> tstore. Both constants are in the fixture.
    # 16384 is `hc_norm`, whose socket is `ds4_hc_norm(float*, float*)` -- TWO arguments and NO
    # gamma. It is handled separately below and must not get the rmsmul treatment.
    ("rmsnorm_engine_1x16384", 16384),
]
EPS = 1e-6


def c(sym, v, ty="i32"):
    return f"    %{sym} = llvm.mlir.constant({v} : {ty}) : {ty}\n"


def mod(name, body):
    return (
        "// Engine slot (ds4-rs-ascend), shape from LAYER0_KERNELS.\n"
        "// Form: llvm.func + hacc.entry + __tile_* -- the shape the emitters consume.\n"
        "module {\n  llvm.func @" + name + "(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) "
        "attributes {hacc.entry} {\n" + body + "\n    llvm.return\n  }\n}\n"
    )


# The engine's `rope_halves`: 64-wide x split into two 32 halves, one 64-wide output
# written as two halves. The chain and the formula are the fixture's
# (ptoas_rs/tests/fixtures/barrier/layer0.pto:364).
ROPE_HALVES = ("rope_halves_engine_1x64", 64, 32)


def rope_halves(slot, width, half):
    body = (
        c("r", 1)
        + c("cw", width)
        + c("ch", half)
        + f'    %lo = llvm.call @__tile_load_f32(%x, %r, %ch) : (!llvm.ptr<1>, i32, i32) -> i32\n'
        f'    %hop = llvm.getelementptr inbounds %x[%ch] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>\n'
        f'    %hi = llvm.call @__tile_load_f32(%hop, %r, %ch) : (!llvm.ptr<1>, i32, i32) -> i32\n'
        f'    %cos = llvm.call @__tile_load_f32(%cs, %r, %ch) : (!llvm.ptr<1>, i32, i32) -> i32\n'
        f'    %sin = llvm.call @__tile_load_f32(%sn, %r, %ch) : (!llvm.ptr<1>, i32, i32) -> i32\n'
        f'    %z = llvm.mlir.constant(0 : i32) : i32\n'
        f'    %t1 = llvm.call @__tile_mul_f32(%z, %lo, %cos, %r, %ch) : (i32, i32, i32, i32, i32) -> i32\n'
        f'    %t2 = llvm.call @__tile_mul_f32(%z, %hi, %sin, %r, %ch) : (i32, i32, i32, i32, i32) -> i32\n'
        f'    %ol = llvm.call @__tile_sub_f32(%z, %t1, %t2, %r, %ch) : (i32, i32, i32, i32, i32) -> i32\n'
        f'    %t3 = llvm.call @__tile_mul_f32(%z, %hi, %cos, %r, %ch) : (i32, i32, i32, i32, i32) -> i32\n'
        f'    %t4 = llvm.call @__tile_mul_f32(%z, %lo, %sin, %r, %ch) : (i32, i32, i32, i32, i32) -> i32\n'
        f'    %oh = llvm.call @__tile_add_f32(%z, %t3, %t4, %r, %ch) : (i32, i32, i32, i32, i32) -> i32\n'
        f'    llvm.call @__tile_store_f32(%o, %ol, %r, %ch) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n'
        f'    %oop = llvm.getelementptr inbounds %o[%ch] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>\n'
        f'    llvm.call @__tile_store_f32(%oop, %oh, %r, %ch) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n'
    )
    return (
        "// Engine slot (ds4-rs-ascend): rope over halves, 4 inputs -> 1 output written\n"
        "// as two 32-element halves. Formula and chain from the layer0 fixture.\n"
        "module {\n  llvm.func @" + slot + "(%x: !llvm.ptr<1>, %cs: !llvm.ptr<1>, "
        "%sn: !llvm.ptr<1>, %o: !llvm.ptr<1>) attributes {hacc.entry} {\n" + body +
        "    llvm.return\n  }\n}\n"
    )


# The engine's quantised projections, served by the CERTIFIED w8a8 kernel.
# `q8mv_kernel_for(d_in, d_out)` in engine_ascend::device launches one row of d_out per slot;
# the deployment package stores them as I8 [d_out, d_in] with a per-output-channel scale.
# Shapes are the engine's own, so these rows serve the slots rather than approximating them.
# Named `proj_*` to match the corpus's existing projection rows: the coverage reader
# identifies a candidate by the KIND token in its stem (`proj`) and then by the extent the
# .tile actually declares, so a row named for the engine slot instead of the kind would
# never be considered however correct it is.
W8A8_MV = [
    ("proj_w8a8_engine_1x1024_k4096", 4096, 1024),
    ("proj_w8a8_engine_1x32768_k1024", 1024, 32768),
    ("proj_w8a8_engine_1x512_k4096", 4096, 512),
    # THE EXPERT SHAPES. `mx4pk_2048_4096` and `mx4pk_4096_2048` are the two slots whose shapes are
    # exactly the MoE expert matvecs (w1/up: 2048 out from k=4096; w2/down: 4096 out from k=2048), and
    # they are PACKED MXFP4. The deployment quantisation is w8a8, so these rows give those shapes a
    # w8a8 artefact -- which is what moving the MoE onto the device needs, since a dispatch through
    # `KernelDispatcher::matvec_f32` finds no binding and falls back to the CPU otherwise.
    ("w8a8_experts_1x2048_k4096", 4096, 2048),
    ("w8a8_experts_1x4096_k2048", 2048, 4096),
    ("w8a8_probe_1x1536_k4096", 4096, 1536),
    # k=8192 uses the CHUNKED arm: the non-chunked one needs eight k-wide buffers and is
    # refused as a UB overflow, and its earlier "successful" emission only fit by aliasing
    # them (25 of 4096 outputs +inf). The chunked form keeps every buffer CHUNK-wide, so k
    # may exceed the unified buffer.
    ("proj_w8a8_engine_1x4096_k8192", 8192, 4096),
    # `hc_mv`: 24 outputs over a 16384-wide activation, from the layer0 fixture's
    # "dst[24] = W_expert . x[16384]". The same chunked arm serves it -- k=16384 needs four
    # chunks -- in the deployment's w8a8 form rather than the fixture's q8_0.
    ("proj_w8a8_engine_1x24_k16384", 16384, 24),
]


def w8a8_mv_chunked(slot, k, n):
    """The chunked form: x is read from GM per chunk, so no k-wide activation tile."""
    body = (
        c("r", 1)
        + c("kk", k)
        + c("nn", n)
        + '    %z = llvm.mlir.constant(0 : i32) : i32\n'
        f'    %res = llvm.call @__tile_matvec_w8a8_gm_f32(%xbuf, %w, %sc, %of, %z, %nn, %kk) : '
        f'(!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32) -> i32\n'
        f'    llvm.call @__tile_store_f32(%o, %res, %r, %nn) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n'
    )
    return (
        "// Engine slot (ds4-rs-ascend): W8A8 matvec, K CHUNKED so k may exceed UB.\n"
        "module {\n  llvm.func @" + slot + "(%xbuf: !llvm.ptr<1>, %w: !llvm.ptr<1>, "
        "%sc: !llvm.ptr<1>, %of: !llvm.ptr<1>, %o: !llvm.ptr<1>) attributes {hacc.entry} {\n"
        + body + "    llvm.return\n  }\n}\n"
    )


def w8a8_mv(slot, k, n):
    body = (
        c("r", 1)
        + c("kk", k)
        + c("nn", n)
        + '    %z = llvm.mlir.constant(0 : i32) : i32\n'
        f'    %x = llvm.call @__tile_load_f32(%xbuf, %r, %kk) : (!llvm.ptr<1>, i32, i32) -> i32\n'
        f'    %res = llvm.call @__tile_matvec_w8a8_f32(%w, %sc, %of, %x, %z, %nn, %kk) : '
        f'(!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32, i32) -> i32\n'
        f'    llvm.call @__tile_store_f32(%o, %res, %r, %nn) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n'
    )
    return (
        "// Engine slot (ds4-rs-ascend): W8A8 matvec, int8 weights with a per-output-channel\n"
        "// scale/offset. Ascend-native deployment quantisation.\n"
        "module {\n  llvm.func @" + slot + "(%xbuf: !llvm.ptr<1>, %w: !llvm.ptr<1>, "
        "%sc: !llvm.ptr<1>, %of: !llvm.ptr<1>, %o: !llvm.ptr<1>) attributes {hacc.entry} {\n"
        + body + "    llvm.return\n  }\n}\n"
    )


def rmsmul_mod(name, body):
    """Three arguments, matching the engine's `ds4_rmsmul_W(x, g, o)`."""
    return (
        "// Engine slot (ds4-rs-ascend): RMS-norm TIMES GAMMA, the engine's `ds4_rmsmul_W`.\n"
        "// x, g, o -- three pointers, because the engine's socket has three.\n"
        "module {\n  llvm.func @" + name + "(%x: !llvm.ptr<1>, %g: !llvm.ptr<1>, "
        "%o: !llvm.ptr<1>) attributes {hacc.entry} {\n" + body + "\n    llvm.return\n  }\n}\n"
    )


# The engine's four quantised projections, in the engine's OWN socket shape:
# `ds4_q8mv_<d_in>_<d_out>(w: si8, s: f32, x: f32, o: f32)` -- four pointers, block scales
# pre-expanded by the host into an f32 plane (layer0.pto:3).
Q8MV = [
    ("proj_q8mv_engine_1x1024_k4096", 4096, 1024),
    ("proj_q8mv_engine_1x32768_k1024", 1024, 32768),
    ("proj_q8mv_engine_1x512_k4096", 4096, 512),
    ("proj_q8mv_engine_1x4096_k8192", 8192, 4096),
]


def q8mv(slot, k, n):
    body = (
        c("r", 1)
        + c("kk", k)
        + c("nn", n)
        + '    %z = llvm.mlir.constant(0 : i32) : i32\n'
        f'    %res = llvm.call @__tile_matvec_q8mv_f32(%w, %s, %x, %z, %nn, %kk) : '
        f'(!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, i32, i32) -> i32\n'
        f'    llvm.call @__tile_store_f32(%o, %res, %r, %nn) : (!llvm.ptr<1>, i32, i32, i32) -> ()\n'
    )
    return (
        "// Engine slot (ds4-rs-ascend): the `ds4_q8mv_<in>_<out>` socket -- four pointers,\n"
        "// block scales pre-expanded into an f32 plane by the host.\n"
        "module {\n  llvm.func @" + slot + "(%w: !llvm.ptr<1>, %s: !llvm.ptr<1>, "
        "%x: !llvm.ptr<1>, %o: !llvm.ptr<1>) attributes {hacc.entry} {\n" + body +
        "    llvm.return\n  }\n}\n"
    )


def kernels():
    out = {}
    for slot, n in SLOTS:
        if n == 16384:
            # hc_norm: pure rms_norm, two arguments (ds4_hc_norm(float*, float*)).
            body = (
                c("r", 1)
                + c("cl", n)
                + f'    %eps = llvm.mlir.constant({EPS} : f32) : f32\n'
                f'    %x = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32\n'
                f'    %o = llvm.call @__tile_rms_norm_f32(%x, %x, %eps, %r, %cl) : (i32, i32, f32, i32, i32) -> i32\n'
                f'    llvm.call @__tile_store_f32(%arg1, %o, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()'
            )
            out[f"rmsnorm_engine_1x{n}"] = mod(f"engine_rms_norm_1_{n}", body)
            continue
        # The engine's socket is `ds4_rmsmul_W(x, gamma, out)`: the fixture's chain ENDS in
        # `tmul(%out, %g)` (layer0.pto:223). A pure rms_norm matches the width and not the
        # computation, which is why these rows carry gamma and three arguments.
        body = (
            c("r", 1)
            + c("cl", n)
            + f'    %eps = llvm.mlir.constant({EPS} : f32) : f32\n'
            f'    %z = llvm.mlir.constant(0 : i32) : i32\n'
            f'    %x = llvm.call @__tile_load_f32(%x, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32\n'
            f'    %gm = llvm.call @__tile_load_f32(%g, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32\n'
            f'    %nrm = llvm.call @__tile_rms_norm_f32(%x, %x, %eps, %r, %cl) : (i32, i32, f32, i32, i32) -> i32\n'
            f'    %sc = llvm.call @__tile_mul_f32(%z, %nrm, %gm, %r, %cl) : (i32, i32, i32, i32, i32) -> i32\n'
            f'    llvm.call @__tile_store_f32(%o, %sc, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()'
        )
        out[f"rmsnorm_engine_rmsmul_1x{n}"] = rmsmul_mod(f"engine_rmsmul_1_{n}", body)
    for slot, k, n in Q8MV:
        out[slot] = q8mv(slot, k, n)
    for slot, k, n in W8A8_MV:
        # the chunked arm is the one that fits k above the buffer
        out[slot] = w8a8_mv_chunked(slot, k, n) if k > 4096 else w8a8_mv(slot, k, n)
    slot, width, half = ROPE_HALVES
    out[slot] = rope_halves(slot, width, half)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, mlir in kernels().items():
        p = OUT / f"{stem}.tile"
        p.write_text(mlir, encoding="utf-8")
        written.append(stem)
    print(f"wrote {len(written)} kernel(s) to {OUT.relative_to(HERE.parent.parent)}:")
    for s in written:
        print("  ", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
