# PackFp4 Pass

Packs frontend logical FP4 nibble types into packed `FP4E2M1X2` for PTOAS.

**Limits, cast policy, and TODOs:** see [FP4](../fp4.md). This page is the
pass algorithm only.

## Overview

Frontend IR counts `pl.FP4` as logical 4-bit nibbles. PTOAS only addresses
`!pto.f4E2M1x2` packed pairs. This pass rewrites dtypes, last-axis covering
sizes / offsets, ND leading strides, and related call operands so later passes
and tile_buf codegen work in packed carrier units. At the PTOAS boundary,
single-device codegen expands GM view / partition last-axis geometry back to
nibble units for pto-isa `GetByteSize`.

**Scope (static-only):** last-axis covering sizes, offsets, and leading strides
must be `ConstInt` (positive even for sizes/strides; even for offsets). Dynamic
last-axis geometry is **rejected**. Prefer UINT8 half-width cache ABI or
hand-written `pl.FP4E2M1X2` — see [FP4](../fp4.md).

**Requires**: `SSAForm`, `NoNestedCalls`, `NormalizedStmtStructure`.

**When to use**: Always in the Default pipeline after `FlattenCallExpr` and
before Outline / `ConvertTensorToTileOps` / `MaterializeTensorStrides`. Run once.

## Unit convention after this pass

After PackFp4, IR and `tile_buf` use **carrier** extents. Single-device codegen
expands only GM `make_tensor_view` / `partition_view` last-axis geometry back to
**nibble** units for pto-isa; the tile side does **not** expand. Full table:
[FP4 unit convention](../fp4.md#unit-convention).

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::PackFp4()` | `passes.pack_fp4()` | Function-level |

```python
from pypto.pypto_core import passes

packed = passes.pack_fp4()(program)
```

## Position in the pipeline

```text
... -> Simplify -> FlattenCallExpr -> PackFp4
    -> OutlineHierarchyScopes -> ... -> ConvertTensorToTileOps -> ...
    -> LegalizeTileCast -> ...
```

## Algorithm

1. **Type rewrite**: Pack every `Tensor` / `Tile` / `Buffer` / `MultiBuffer`
   that still carries logical FP4 into `FP4E2M1X2`. `DistributedTensor` FP4 and
   scalar FP4 are **rejected** (see [FP4](../fp4.md)).
2. **Last-axis covering sizes** (`shape`, `valid_shape`, slice / load sizes):
   positive even `ConstInt` → `K / 2`.
3. **Last-axis offsets**: even `ConstInt` → `offset / 2`.
4. **ND leading strides**: contiguous last axis (`stride[-1] == 1`); leading
   strides pack under covering-size rules.
5. **Call operands**: rewrite shape / offset / size tuples on the whitelist
   (slice, load, store, assemble, **view** including optional `valid_shape`,
   create, full, …). **Reject** reshape / transpose / read / write / remote /
   window / put / get. Unlisted ops that pass a rank>0 `TupleType` coordinate
   fail loudly.
6. **Layout guards**: reject DN / NZ tensors, col-major tiles, and cube memory.

## Example

**Before** (logical FP4, last dim 64 nibbles):

```python
x: pl.Tensor[[16, 64], pl.FP4]
tile = pl.load(x, [0, 2], [16, 32])
```

**After PackFp4**:

```python
x: pl.Tensor[[16, 32], pl.FP4E2M1X2]
tile = pl.load(x, [0, 1], [16, 16])
```
