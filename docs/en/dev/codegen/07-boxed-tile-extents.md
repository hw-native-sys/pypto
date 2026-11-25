# Boxed Tile Extents

Every `pto.alloc_tile` PyPTO emits is validated against the box grid PTOAS will
check it against. PTO addresses a boxed tile one box at a time, so a tile whose
*physical* extent is not a whole number of boxes has no address at all.

The rule mirrors PTOAS' `verifyBoxedTileLayout` exactly:

| Layout | Box (rows x cols) |
| ------ | ----------------- |
| fractal 1024 (`Acc`) | 16 x 16 |
| fractal 512, `slayout = row_major` (`Mat` / `Left`) | 16 x (32 / sizeof(dtype)) |
| fractal 512, `slayout = col_major` (`Right`, the transposed dual) | (32 / sizeof(dtype)) x 16 |
| `slayout = none_box` | not boxed — no rule |

with PTOAS' own exemptions kept: the *row* rule is skipped for `Vec` and for a
single-row tile (the NZ map degenerates there), while the column rule always
applies. The MX-scale fractal and sub-byte carriers are left to PTOAS, which
diagnoses them itself.

**Why here rather than in PTOAS.** PTOAS rejects the same shape, but its message
names its own internals and offers no remedy:

```text
'pto.alloc_tile' op expects result boxed tile rows to be a multiple of innerRows (16), but got 100
```

Raising at the emission site instead reports the tile, the axis, the extent to
reach, and how to reach it:

```text
a Mat tile of physical shape [100, 128] and dtype fp16 must be a whole number of
16x16 fractal boxes, but its row extent 100 is not a multiple of 16. PTO addresses
a boxed tile one box at a time, so a partial box has no address. The *logical*
extent is free -- allocate 112 on that axis and declare 100 as the tile's
valid_shape (`valid_shape=` on pl.load / pl.tile.create + pl.set_validshape),
which moves and computes only the real data. A tensor-level pl.matmul /
pl.matmul_acc does this for its M axis automatically.
```

`ComputeAllocTileFields` is the single choke point every allocation passes
through — the per-variable declaration, the hoisted `extra_alloc_tiles`, and the
control-flow paths alike — so the check sees exactly what is emitted and cannot
drift from it. A tensor-level `pl.matmul` / `pl.matmul_acc` never trips it *on
its M axis*, which is boxed for the user in
[`ConvertTensorToTileOps`](../passes/13-convert_tensor_to_tile_ops.md#cube-operand-m-axis-boxing).
The axes that remain the user's responsibility are `K` and `N`.
