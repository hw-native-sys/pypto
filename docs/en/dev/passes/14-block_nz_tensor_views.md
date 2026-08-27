# BlockNzTensorViews Pass

## Overview

`BlockNzTensorViews` turns a *logical* `pl.NZ` tensor into the *blocked* form
pto-isa's `Layout::NZ` GlobalTensor requires, and retargets the `tile.load`
that reads it.

A `pl.Tensor[[..., R, C], dtype, pl.NZ]` annotation is an **assertion about the
bytes already in GM**: they are stored in PTO-native NZ fractal order. It is not
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
shape   = [..., C/c0, R/16, 16, c0]
strides = [..., C*R,  R*c0, 16*c0, c0, 1]
```

Reading the shape from the inside out: `c0` contiguous elements form one 32-byte
C0 line, 16 rows form one `16 x c0` fractal (512 bytes), `R/16` fractals walk
down the row axis, and the **outermost** dim steps between column blocks. That
is "column blocks outside, row fractals inside" — the same byte order the tile
side expresses as `blayout=col_major, slayout=row_major, fractal=512`.

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
`BuildRowMajorStrides` path as ND. `MaterializeTensorStrides` (pass 31) fills the
stride later; this pass only rewrites the shape.

This amends RFC #1300's claim that NZ has "no logical-stride representation" —
true of a logical 2-D shape, false of the blocked rank-(r+2) one.

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

**Phase 2 — retarget the consuming `tile.load`.**

```text
# before  (slicing w[1:2, 256:512, 512:1024] out of an [E, N, K] weight)
wt: pl.Tile[[256, 512], pl.INT8, pl.Mem.Mat] =
    pl.tile.load(w, [1, 256, 512], [1, 256, 512], target_memory=pl.Mem.Mat)

# after   (offsets -> [.., k0/c0, n0/16, 0, 0];  sizes -> blocked)
wt: pl.Tile[[256, 512], pl.INT8, pl.Mem.Mat] =
    pl.tile.load(w, [1, 16, 16, 0, 0], [1, 16, 16, 16, 32], target_memory=pl.Mem.Mat)
```

The two trailing offsets must be **constants**. Milestone 1 performs the
`k0/c0` / `n0/16` mapping only on `ConstInt`, so a symbolic offset is rejected
even when it is provably aligned — mapping one would need a divisibility proof
plus an algebraic rewrite (`nb*256` -> `nb*16`), which is not implemented.

A loop-derived slice is therefore **not supported yet**:

```python
for nb in pl.spmd(N // N_TILE):
    n0 = nb * N_TILE
    wt = w[n0 : n0 + N_TILE, 0:K_TILE]   # rejected: dynamic offset on shape[-2]
```

This is the main gap between this milestone and the grouped-matmul weight path
that motivated it. Tracked as issue #2548.

The destination `TileType` is **preserved verbatim**: the GM partition becomes
rank-(r+2), the tile stays the logical 2-D operand. The load is therefore rebuilt
with the explicit-type `Call` constructor rather than `OpRegistry::Create`, which
would re-deduce a rank-(r+2) tile from the blocked shapes argument.

After this pass no logical-shaped NZ `TensorType` survives, so nothing
downstream needs to know NZ is special — including codegen, which derives the
rank of `pto.make_tensor_view`, its `!pto.tensor_view<>` type and the
`pto.partition_view` independently from `TensorType::shape_` and must see them
agree.

## Generated code

```mlir
%w_view = pto.make_tensor_view %arg1,
    shape = [%c16, %c16, %c16, %c32], strides = [%c8192, %c512, %c32, %c1]
    {layout = #pto.layout<nz>} : !pto.tensor_view<?x?x?x?xi8>
%w_pview = pto.partition_view %w_view,
    offsets = [%c0, %c0, %c0, %c0], sizes = [%c16, %c16, %c16, %c32]
    : !pto.tensor_view<?x?x?x?xi8> -> !pto.partition_tensor_view<16x16x16x32xi8>
pto.tload ins(%w_pview : !pto.partition_tensor_view<16x16x16x32xi8>)
          outs(%wt : !pto.tile_buf<loc=mat, dtype=i8, rows=256, cols=512,
                                   blayout=col_major, slayout=row_major, fractal=512, ...>)
```

## Scope and rejections

Milestone 1 is deliberately narrow. Everything outside it is rejected with a
diagnostic naming the fix — an NZ tensor must never be silently mis-addressed.

| Condition | Outcome |
| --------- | ------- |
| `shape[-2] % 16 != 0` | rejected — a partial fractal has no representation |
| `shape[-1] % c0 != 0` | rejected — a partial C0 line has no representation |
| dynamic `shape[-2]` / `shape[-1]` | rejected — divisibility cannot be proven |
| slice offset not fractal-aligned | rejected — no blocked representation |
| **dynamic (non-constant) trailing slice offset** | **rejected — only `ConstInt` offsets are mapped; a provably aligned symbolic offset is refused too (#2548)** |
| rank < 2 | rejected |
| `target_memory != Mat` (or absent) | rejected — NZ→NZ is the cube operand path |
| consumer other than `tile.load` | rejected — NZ is read-only here |
| explicit stride or partial `valid_shape` | rejected |
| distributed tensor | rejected — `remote_load` has no NZ blocking |
| `tensor.view` / `tensor.reinterpret_view` of NZ | rejected at op construction |

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

## Downstream dependency

PTOAS infers a `make_tensor_view`'s layout structurally. Blocked NZ and ND are
structurally identical (both row-major), so PTOAS currently infers `nd` and
overrides the explicit `nz` annotation, failing with
`layout mismatch: user-specified layout=nz but inferred=nd`. The failure is
safe rather than silent: with the blocked shape, pto-isa's ND→NZ `TLOAD` path
requires `staticShape[0..2] == 1`, which the blocked dims violate, so the
generated C++ fails a `static_assert` instead of computing wrong results.
End-to-end use waits on PTOAS trusting the explicit annotation.

## Related

- [13-flatten_tile_nd_to_2d.md](13-flatten_tile_nd_to_2d.md) — skips its ND2NZ window collapse for NZ sources
- [31-materialize_tensor_strides.md](31-materialize_tensor_strides.md) — fills the blocked NZ stride
- [../ir/02-types.md](../ir/02-types.md) — `TensorLayout` and `TensorView`
