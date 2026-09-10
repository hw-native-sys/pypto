# BlockNzTensorViews Pass

## Overview

`BlockNzTensorViews` turns a *logical* `pl.NZ` tensor into the *blocked* form
pto-isa's `Layout::NZ` GlobalTensor requires, and retargets the `tile.load`
that reads it.

A `pl.Tensor[[R, C], dtype, pl.NZ]` annotation — or `[[B, R, C], ...]` — is an
**assertion about the bytes already in GM**: they are stored in PTO-native NZ
fractal order. It is not
a request to convert anything. The DSL keeps the logical shape and logical
slicing; this pass supplies the physical description the backend needs.

The payoff is that a matmul B-operand `TLOAD` becomes NZ→NZ instead of ND→NZ,
which removes the online fractal conversion from every weight load.

## The blocked form

With `c0` = the number of elements in a 32-byte C0 line (`256 / dtype bits`; 32 for
`INT8`) and a 16-row fractal, pto-isa
describes an NZ buffer as (`pto/common/pto_tile.hpp`, `TileShape2D` /
`BaseShape2D` specialisations for `Layout::NZ`):

```text
shape   = [B, C/c0, R/16, 16, c0]
strides = [C*R, R*c0, 16*c0, c0, 1]
```

Reading the shape from the inside out: `c0` contiguous elements form one 32-byte
C0 line, 16 rows form one `16 x c0` fractal (512 bytes), `R/16` fractals walk
down the row axis, and `C/c0` steps between column blocks. That is "column blocks
outside, row fractals inside" — the same byte order the tile side expresses as
`blayout=col_major, slayout=row_major, fractal=512`.

**The rank is fixed at 5, not `logical rank + 2`.** The leading batch slot `B`
exists whether or not the logical tensor has a leading axis, so a logical rank-2
`[N, K]` weight blocks to `[1, K/c0, N/16, 16, c0]` with the batch materialised
as `1`. PTOAS checks this arity directly — a rank-4 view is refused with
`'pto.make_tensor_view' op user-specified layout=nz requires a rank-5 view`, no
matter how self-consistent the surrounding IR is.

### Why NZ needs no stride rule of its own

Row-major strides over the blocked shape *are* pto-isa's NZ strides:

| slot | row-major derivation | `BaseShape2D<T, R, C, NZ>` |
| ---- | -------------------- | -------------------------- |
| `c0` | `1` | `1` |
| `16` | `c0` | `C0Size` |
| `R/16` | `16*c0` | `FRACTAL_NZ_ROW*C0Size` |
| `C/c0` | `(R/16)*16*c0 = R*c0` | `rows*C0Size` |
| leading | `(C/c0)*R*c0 = C*R` | `cols*rows` |

So once the shape is blocked, NZ is an ordinary member of the row-major family
and `BuildLogicalStridesFromLayout` handles it through the same
`BuildRowMajorStrides` path as ND. `MaterializeTensorStrides` (pass 33) fills the
stride later; this pass only rewrites the shape.

This amends RFC #1300's claim that NZ has "no logical-stride representation" —
true of a logical 2-D shape, false of the blocked rank-5 one.

## Position in the pipeline

```text
... -> LowerCompositeOps -> FlattenTileNdTo2D -> BlockNzTensorViews -> LegalizeTileCast -> ...
```

Three constraints fix this slot:

- **After `ConvertTensorToTileOps` / `LowerCompositeOps`** — the `tile.load` ops
  phase 2 rewrites must already exist.
- **After `FlattenTileNdTo2D`** — the destination tile must already be the
  logical 2-D operand. Blocking a still-ND-rank tile would leave a `tile.load`
  whose type annotation and argument ranks cannot both be printed, breaking the
  printer round-trip.
- **Before `MaterializeTensorStrides`** — which asserts every NZ view is blocked
  and then fills its row-major stride.

`FlattenTileNdTo2D` skips its ND2NZ source-window collapse for an NZ source
(that collapse exists because ND→NZ needs a 2-D GlobalTensor; NZ→NZ does not),
so the logical window is still intact when this pass runs.

## Behavior

**Phase 1 — block every NZ `TensorType` shape.**

```text
# before
w: pl.Tensor[[32, 2048, 4096], pl.INT8, pl.NZ]

# after  (c0 = 32:  4096/32 = 128,  2048/16 = 128)
w: pl.Tensor[[32, 128, 128, 16, 32], pl.INT8, pl.NZ]
```

A logical rank-2 weight lands on the same rank, with the batch synthesised:

```text
# before
w: pl.Tensor[[256, 512], pl.INT8, pl.NZ]

# after  (c0 = 32:  512/32 = 16,  256/16 = 16;  batch materialised as 1)
w: pl.Tensor[[1, 16, 16, 16, 32], pl.INT8, pl.NZ]
```

**Phase 2 — retarget the consuming `tile.load`.**

```text
# before  (slicing w[1:2, 256:512, 512:1024] out of an [E, N, K] weight)
wt: pl.Tile[[256, 512], pl.INT8, pl.Mem.Mat] =
    pl.tile.load(w, [1, 256, 512], [1, 256, 512], target_memory=pl.Mem.Mat)

# after   (offsets -> [.., k0/c0, n0/16, 0, 0];  sizes -> blocked)
wt: pl.Tile[[256, 512], pl.INT8, pl.Mem.Mat] =
    pl.tile.load(w, [1, 16, 16, 0, 0], [1, 16, 16, 16, 32], target_memory=pl.Mem.Mat)
```

### Symbolic trailing offsets

A trailing offset does not have to be a constant, but its alignment must be
**proven** — never assumed. An offset arrives here as the SSA name it was bound
to, not as the arithmetic that produced it, so `IsProvableMultipleOf`
(`tensor_view_semantics.h`) walks two kinds of binding the pass collects from
the enclosing function in one read-only sweep:

| Binding | Multiple of the axis factor | Non-negative |
| ------- | --------------------------- | ------------ |
| `AssignStmt` (`n0 = nb * 256`) | one factor of the product is a multiple | both factors are non-negative |
| `ForStmt` (`for k0 in pl.pipeline(512, 4096, 512)`) | `start` and `step` are both multiples | `start` and `step` are both non-negative |
| `tile.get_block_idx` / `tile.get_block_num` | — | a lane number is never negative |
| `ConstInt` | the value is a multiple | the value is `>= 0` |

Sums and products compose from those; a difference proves divisibility but never
its sign, so it is refused. **Both** columns must hold — see [Why the sign is
proven too](#why-the-sign-is-proven-too). The grouped-matmul weight path that
motivated the feature therefore compiles:

```python
for nb in pl.spmd(N // N_TILE):
    n0 = nb * N_TILE                     # -> row fractal offset  n0 // 16
    for k0 in pl.pipeline(K_TILE, K, K_TILE, stage=2):
        wt = w[n0 : n0 + N_TILE, k0 : k0 + K_TILE]   # -> c0 block offset  k0 // c0
```

Anything the walk cannot prove is rejected with a diagnostic naming the provable
forms. That refusal is the design: an NZ tensor addressed from a guessed
coordinate reads the wrong fractal, and nothing downstream would notice.

#### Why the offset is divided, not re-associated

Only the *result* of the offset is divided (`FloorDiv(offset, divisor)`). The
tempting rewrite — turning `n0 = nb * 256` into `nb * 16` and saving the runtime
division — is unsound, because IR arithmetic wraps at its declared width while
the rewritten form does not:

```text
x : INT32 = 1 << 24
(x * 256) / 16   ==  0            # x * 256 wraps to 0 in i32
x * (256 / 16)   ==  268435456    # re-associated: no wrap, wrong fractal
```

Widening is not the culprit, and rebuilding at the original width does not help:
for `a * b = q * 2^W + r`, the original yields `r / d` while any re-association
yields `r / d + q * 2^(W - log2 d)`. Dividing the offset as a whole keeps its
own dtype and overflow behaviour intact.

The proof itself survives wraparound because every divisor here is a power of
two and so divides `2^W`: reducing a multiple of `d` modulo `2^W` leaves a
multiple of `d`. An `INTERNAL_CHECK` pins that precondition.

One further limit is deliberate: an `IterArg` is never resolved, because its
value changes every iteration, so neither its initial value nor any binding
recorded for it describes the value a given use sees.

#### Why the sign is proven too

Divisibility alone does not make a coordinate safe. `n0 = -16` is a clean
multiple of 16, and `FloorDiv(n0, 16)` is `-1`; codegen then *clamps* a negative
`pto.partition_view` offset to 0 instead of failing, so the load reads fractal 0
and returns silently wrong data — the same shape [#2543] fixed for row indices.

The constant path already refuses a negative literal, so proving only
divisibility for a symbolic offset would mean the identical value is caught
written inline and waved through once bound to a name.
`IsProvableNonNegative` closes that gap.

[#2543]: https://github.com/hw-native-sys/pypto/pull/2543

The destination `TileType` is **preserved verbatim**: the GM partition becomes
rank-5, the tile stays the logical 2-D operand. The load is therefore rebuilt
with the explicit-type `Call` constructor rather than `OpRegistry::Create`, which
would re-deduce a rank-5 tile from the blocked shapes argument.

After this pass no logical-shaped NZ `TensorType` survives, so nothing
downstream needs to know NZ is special — including codegen, which derives the
rank of `pto.make_tensor_view`, its `!pto.tensor_view<>` type and the
`pto.partition_view` independently from `TensorType::shape_` and must see them
agree.

## Generated code

For the logical rank-2 `w: pl.Tensor[[256, 512], pl.INT8, pl.NZ]` above — note
the leading `%c1` / `%c131072` / `1x` carrying the synthesised batch through all
three sites:

```mlir
%w_view = pto.make_tensor_view %arg1,
    shape = [%c1, %c16, %c16, %c16, %c32],
    strides = [%c131072, %c8192, %c512, %c32, %c1]
    {layout = #pto.layout<nz>} : !pto.tensor_view<?x?x?x?x?xi8>
%w_pview = pto.partition_view %w_view,
    offsets = [%c0, %c0, %c0, %c0, %c0], sizes = [%c1, %c16, %c16, %c16, %c32]
    : !pto.tensor_view<?x?x?x?x?xi8> -> !pto.partition_tensor_view<1x16x16x16x32xi8>
pto.tload ins(%w_pview : !pto.partition_tensor_view<1x16x16x16x32xi8>)
          outs(%wt : !pto.tile_buf<loc=mat, dtype=i8, rows=256, cols=512,
                                   blayout=col_major, slayout=row_major, fractal=512, ...>)
```

which **PTOAS 0.61 and later** turns into a real NZ `GlobalTensor`:

```cpp
GlobalTensor<int8_t, pto::Shape<1, 16, 16, 16, 32>,
             pto::Stride<131072, 8192, 512, 32, 1>, pto::Layout::NZ> ...;
```

On 0.60 and earlier the same IR does not assemble — see
[Assembler version](#assembler-version) below.

## Scope and rejections

Milestone 1 is deliberately narrow. Everything outside it is rejected with a
diagnostic naming the fix — an NZ tensor must never be silently mis-addressed.

| Condition | Outcome |
| --------- | ------- |
| `shape[-2] % 16 != 0` | rejected — a partial fractal has no representation |
| `shape[-1] % c0 != 0` | rejected — a partial C0 line has no representation |
| dynamic `shape[-2]` / `shape[-1]` | rejected — divisibility cannot be proven |
| slice offset not fractal-aligned | rejected — no blocked representation |
| symbolic trailing slice offset, alignment and sign both provable | mapped — see [Symbolic trailing offsets](#symbolic-trailing-offsets) |
| symbolic trailing slice offset, alignment not provable | rejected — never divided on the assumption that it is aligned |
| symbolic trailing slice offset, sign not provable | rejected — a negative offset is clamped, not caught, at the partition view |
| logical rank 2 | blocked to `[1, C/c0, R/16, 16, c0]` — batch materialised |
| logical rank 3 | blocked to `[B, C/c0, R/16, 16, c0]` — leading axis is the batch |
| logical rank < 2 | rejected — the trailing pair is the fractal plane |
| logical rank > 3 | rejected — one batch slot cannot hold two leading axes (see below) |
| `target_memory != Mat` (or absent) | rejected — NZ→NZ is the cube operand path |
| consumer other than `tile.load` | rejected — NZ is read-only here |
| explicit stride or partial `valid_shape` | rejected |
| distributed tensor | rejected — `remote_load` has no NZ blocking |
| `tensor.view` / `tensor.reinterpret_view` of NZ | rejected at op construction |

### Why logical rank 4+ is rejected

pto-isa's NZ `GlobalTensor` has exactly **one** batch slot, so a logical
`[G, E, N, K]` weight would have to fold its two leading axes into it. That fold
is sound on the *shape* — a dense row-major tensor's leading strides collapse
exactly, `G*E` with stride `C*R` — but not on the *offsets*: a slice `w[g, e, ...]`
would need the coordinate re-associated into `g*E + e`, which is precisely the
arithmetic `BlockNzOffsets` refuses to invent (see [Symbolic trailing
offsets](#symbolic-trailing-offsets) for why re-association is unsound in
general). Rejecting names the restriction at the annotation; the alternative is a
view PTOAS refuses while naming SSA the user never wrote.

Reshape to `[B, R, C]` before the NZ annotation, or annotate the tensor as
`pl.ND`.

Sub-byte dtypes (INT4 / UINT4 / FP4 / HF4 / BOOL) are rejected as a **PyPTO
milestone-1 scope limit, not a hardware one** — pto-isa's NZ machinery does
handle FP4 (`tload_common.hpp` carries explicit `caps::IsFP4` branches and
asserts `staticShape[4] == C0_SIZE_BYTE / sizeof(DType)`). `c0` is already
derived from the bit width, so the arithmetic is ready when the packed-nibble
addressing is validated end to end.

The alignment diagnostics are user-facing (`CHECK_SPAN` → `ValueError`) and live
in `BlockNzShape`. Downstream, an *unblocked* NZ view is a pass-ordering
invariant instead (`INTERNAL_CHECK_SPAN`), enforced by `CheckNzViewIsBlocked` in
`MaterializeTensorStrides` and by the `TensorViewCanonical` verifier.

## Idempotence

Blocking is not idempotent — blocking a blocked shape would be wrong — and the
structural `IsBlockedNzShape` test cannot distinguish a blocked shape from a
logical one that merely ends in `[16, c0]`. The pass therefore stamps
`nz_tensor_views_blocked` on each function it rewrites and returns early when it
sees that attribute.

## Assembler version

Whether this pass's output assembles at all depends on the PTOAS release, and
the repository's pinned version is not yet the one that works.

| PTOAS | Behavior |
| ----- | -------- |
| ≤ 0.60 | Infers the layout structurally. Blocked NZ and ND are structurally identical (both row-major), so it infers `nd`, overrides the explicit `nz` annotation, and fails with `layout mismatch: user-specified layout=nz but inferred=nd`. No NZ view assembles, at any rank. |
| ≥ 0.61 | Treats an explicit `ND` / `DN` / `NZ` annotation as authoritative and validates it, so the descriptor above assembles. It also enforces NZ's arity directly: a view of any rank but 5 is refused with `'pto.make_tensor_view' op user-specified layout=nz requires a rank-5 view`. |

`toolchain/versions.env` currently pins **v0.60**, so `pl.NZ` does not yet work
end to end on the pinned toolchain, and no test in the tree reaches the
assembler with an NZ tensor. To exercise the path, point `PTOAS_ROOT` at a 0.61
or later install.

The 0.60 failure is safe rather than silent in both directions: it stops at the
layout mismatch above, and even past that, pto-isa's ND→NZ `TLOAD` path requires
`staticShape[0..2] == 1`, which the blocked dims violate — so the generated C++
fails a `static_assert` instead of computing wrong results.

## Related

- [14-flatten_tile_nd_to_2d.md](14-flatten_tile_nd_to_2d.md) — skips its ND2NZ window collapse for NZ sources
- [32-materialize_tensor_strides.md](33-materialize_tensor_strides.md) — fills the blocked NZ stride
- [../ir/02-types.md](../ir/02-types.md) — `TensorLayout` and `TensorView`
