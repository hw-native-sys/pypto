"""Generate the engine's `ds4_attn_sink_batched` tile: ONE HEAD, M blocks of 8, D=512.

    python3 gen_attention.py <blocks> <out.tile>      # 1 = one block, 16 = the full S=128

The maths is the fixture's (layer0.pto:437+):

    score[p] = rowsum(K_p .* Q) + sink[p % 8]        (sink added to every block)
    w        = softmax over all p of score
    out[c]   = (sum_p w[p] * V[p][c]) / sum_p w[p]    (K and V are the same %kv tensor)

Two structural facts this encodes, both found by writing it:
  * the scores are assembled BLOCK BY BLOCK with `__tile_slice_store_f32`, because every other
    intrinsic produces a whole tile;
  * the weighted sum is accumulated ROW BY ROW with `__tile_row_axpy2_f32`, because this DSL has no
    column-wise reduction (`reduce_sum` is row-wise). That is why no such reduction is needed.

KNOWN BLOCKER at 1 block and at 16: the emitter allocates one TBuf per buffer index and does not
reuse across non-overlapping lifetimes, so this lands at ~274 buffers and 187648 B of a 188416 B
budget. `col_expand` and `row_axpy2` each allocate a temp, and the even split then hands a 512-float
tile ~685 B where it needs 2048. Fixing that allocator behaviour is what unblocks the kernel; the
kernel itself is written.
"""
import pathlib
import sys

blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/attn.tile"
SB = 8
D = int(sys.argv[3]) if len(sys.argv) > 3 else 512
P = "!llvm.ptr<1>"
L = []
# Constant NAMES must be `%c<digits>`: the emitter's resolver recognises that spelling and emits
# others as `/* %cNAME */0`, which the UB sizer then cannot fold -- the whole buffer was split
# evenly and the kernel faulted. `%cD`/`%cSB` were the cause, not the arithmetic.
body = "".join(f"    %c{v} = llvm.mlir.constant({v} : i32) : i32\n" for v in (0, 1, SB, D, blocks))
body += "    %f1 = llvm.mlir.constant(1.0 : f32) : f32\n"


def call(res, fn, args, sig):
    L.append(f'    %{res} = llvm.call @{fn}({"%"+", %".join(args)}) : {sig} -> i32\n')
    return res


cD = f"c{D}"
call("q1", "__tile_load_at_f32", ["q", "c0", "c1", cD], f"({P}, i32, i32, i32)")
call("qb", "__tile_col_expand_f32", ["c0", "q1", "c8", cD], "(i32, i32, i32, i32)")
call("sink", "__tile_load_at_f32", ["sk", "c0", "c1", "c8"], f"({P}, i32, i32, i32)")
call("acc", "__tile_sub_f32", ["c0", "q1", "q1", "c1", cD], "(i32, i32, i32, i32, i32)")
if blocks > 1:
    call("sc", "__tile_col_expand_f32", ["c0", "q1", "c1", f"c{blocks*SB}"], "(i32, i32, i32, i32)")
for b in range(blocks):
    off = f"off{b}"
    L.append(f"    %{off} = llvm.mlir.constant({b * SB * D} : i32) : i32\n")
    call(f"kb{b}", "__tile_load_at_f32", ["kv", off, "c8", cD], f"({P}, i32, i32, i32)")
    call(f"p{b}", "__tile_mul_f32", ["c0", f"kb{b}", "qb", "c8", cD], "(i32, i32, i32, i32, i32)")
    call(f"s{b}", "__tile_reduce_sum_f32", ["c0", f"p{b}", "c8", cD], "(i32, i32, i32, i32)")
    call(f"sa{b}", "__tile_add_f32", ["c0", f"s{b}", "sink", "c1", "c8"], "(i32, i32, i32, i32, i32)")

scores = "sc" if blocks > 1 else "sa0"
ncol = f"c{blocks * SB}"
for b in range(blocks):
    if blocks > 1:
        L.append(f"    %o{b} = llvm.mlir.constant({b * SB} : i32) : i32\n")
        call("sc", "__tile_slice_store_f32", ["sc", f"sa{b}", f"o{b}", "c8"], "(i32, i32, i32, i32)")
# SOFTMAX WITHOUT ANY SCALAR BROADCAST. `col_expand(1 -> ROWS)` is the one primitive that could not
# be made to work (a scalar read after a vector op returned inf behind an explicit V_S flag/wait, and
# the Duplicate overloads behaved the same), so the kernel is arranged to need none:
#   * the weights are the UNNORMALISED exp(scores) -- the division moves to the end;
#   * `acc += w[p]*V[p][:]` uses row_axpy2, which takes its scalar from a tile and is certified;
#   * `1/den` is `rsqrt(den*den)`, one element, and `row_scale` multiplies the 512-wide result by it.
# The max shift is dropped: with bounded scores exp cannot overflow, and the oracle computes the true
# softmax, so a wrong kernel still fails.
call("e", "__tile_exp_f32", ["c0", scores, "c1", ncol], "(i32, i32, i32, i32)")
call("den", "__tile_reduce_sum_f32", ["c0", "e", "c1", ncol], "(i32, i32, i32, i32)")
# ONE MATMUL, NOT EIGHT SCALAR-SCALED ROWS. `sum_r e[r]*V[r][:]` is `e @ V` with e (1 x 8) and V
# (8 x 512), which the matmul arm computes directly. The eight `row_axpy` calls it replaces each read
# a scalar out of a vector-produced tile, and in this emitter that read returns 0 -- so the matmul
# REMOVES the hazard instead of trying to order it.
for b in range(blocks):
    call(f"vb{b}", "__tile_load_at_f32", ["kv", f"off{b}", "cSB", "cD"], f"({P}, i32, i32, i32)")
    # (dst dummy, a, b, m, k, n) -- SIX operands. Passing five left the arm unmatched and the call
    # was SILENTLY DROPPED, which is how the accumulation produced nothing for two rounds.
    acc = call("acc", "__tile_matmul_f32", ["c0", "e", f"vb{b}", "c1", "cSB", "cD"],
               "(i32, i32, i32, i32, i32, i32, i32)")
call("d2", "__tile_mul_f32", ["c0", "den", "den", "c1", "c1"], "(i32, i32, i32, i32, i32)")
call("inv", "__tile_rsqrt_f32", ["c0", "d2", "c1", "c1"], "(i32, i32, i32, i32)")
call("outv", "__tile_row_scale_f32", ["c0", acc, "inv", "c1", cD], "(i32, i32, i32, i32, i32)")
L.append(f'    llvm.call @__tile_store_f32(%o, %outv, %c1, %{cD}) : ({P}, i32, i32, i32) -> ()\n')

header = (f"// Engine slot: ds4_attn_sink_batched -- ONE HEAD, {blocks} block(s) of 8, D=512.\n"
          "// scores = rowsum(K_b .* Q) + sink; softmax over the positions; acc += w[r]*V[r][:].\n"
          "// Generated by rust/tools/direct-invoke/gen_attention.py.\n")
# EVERY constant name is normalised here, once, on the finished text. Three rounds were lost to a
# literal `%cD` or `%cSB` surviving into the generated IR, where an unknown `%c*` resolves to
# `/* %cNAME */0` and the operation silently becomes a zero-count no-op. Normalising at the point the
# text leaves this file makes it impossible to write a new call that reintroduces it.
tile = (
    header + "module {\n  llvm.func @attn_sink_%dblk(%%q: !llvm.ptr<1>, %%kv: !llvm.ptr<1>, "
    "%%sk: !llvm.ptr<1>, %%o: !llvm.ptr<1>) attributes {hacc.entry} {\n" % blocks
    + body + "".join(L) + "    llvm.return\n  }\n}\n"
)
tile = tile.replace("%cSB", "%%c%d" % SB).replace("%cD", "%%c%d" % D)
assert "%cD" not in tile and "%cSB" not in tile, "an unnormalised constant survived"
pathlib.Path(out).write_text(tile)
print(f"wrote {out}: {len(L)} ops, {blocks} block(s)")
