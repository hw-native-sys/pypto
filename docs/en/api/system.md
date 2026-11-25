# `pl.system`

Synchronization, cache and cross-core primitives. These are side-effecting statements
rather than value-producing operators — most return nothing.

Use `pl.system.barrier(pipe=...)` with any `pl.PipeType` member for a pipeline
barrier. A barrier is pipe-scoped: it drains whatever was issued on that pipe,
so the pipe a transfer rides decides which barrier orders it — `MTE2` the
GM → UB/L1 loads, `MTE3` the UB → GM stores and the A5 UB → L1 route, `MTE1` the
L1 → L0A/L0B operand feed, and `FIX` the L0C writeback. `ALL` covers all
pipelines on the current core; cross-core arrival uses `pl.system.syncall(...)`.
The old `bar_v()`, `bar_m()`, and `bar_all()` APIs have been removed.
Directional event handoffs still use a separate
`set_flag` / `wait_flag` pair; see [pipeline ordering](../user/language/04-scopes.md#cross-lane-ordering-is-yours-too).

## Explicit AIV lane

`tpush_to_aic` and `tpop_from_aic` accept the optional keyword
`aiv_subblockid`. A Python integer 0 or 1 becomes an INT64 constant; a dynamic
operand must have type `pl.Scalar[pl.INT64]`. Cast INDEX or other scalar dtypes
explicitly with `pl.cast(value, pl.INT64)`. The lane is an SSA operand, so it
survives outlining and kernel splitting. Omitting it preserves the backend's
physical lane selection and existing pipe emission.

Inside an AIV kernel with a bidirectional pipe already initialized:

```python
lane = pl.cast(pl.tile.get_subblock_idx(), pl.INT64)
outgoing = pl.tile.full([16, 16], dtype=pl.FP32, value=1.0)
pl.system.tpush_to_aic(outgoing, split=1, aiv_subblockid=lane)
received = pl.system.tpop_from_aic(
    shape=[16, 16], dtype=pl.FP32, split=1, aiv_subblockid=lane
)
# Consume received before releasing its C2V slot.
pl.system.tfree_to_aic(received)
```

Keep `outgoing` live until the AIC consumer releases its V2C slot with
`tfree_to_aiv`. The C2V release shown here only releases `received`.

Only AIV-side Vec Tile entries and split codes 1/2/3/4 accept the operand.
No-split (`split=0`), GlobalTensor entries, and AIC-side `tpush_to_aiv` /
`tpop_from_aiv` do not. Existing split, shape, layout and initialization
contracts still apply; this argument does not create a pipe or change its
schema. In particular, PyPTO continues to reject odd V2C push codes 3/4.
The matrix describes compiler contracts; device execution needs separate
validation for each architecture.

| Path | A3 | A5 |
| ---- | -- | -- |
| Explicit lane, Vec push/pop, split 1/2 | PTOAS operand supported | PTOAS operand supported; V2C layout adapter retained |
| Explicit lane, C2V pop, split 3/4 | Existing GM-backed odd split rules apply | Existing GM-backed odd split rules apply |
| Explicit lane, V2C push, split 3/4 | PyPTO restriction retained | Backend rejects odd V2C |
| No-split / GlobalTensor / AIC-side lane | Rejected | Rejected |

At runtime the author must keep dynamic lanes in `{0, 1}`, provide each logical
lane exactly as required by the matching split protocol, and pair push/pop/free
counts and pipe IDs. No runtime range guard is inserted. A constant outside
that range is rejected. Selecting a logical lane does not move the operation
to another physical core or change the executing kernel.

The effect follows the backend transport. A5's direct Acc→UB C2V pipe returns
the executing AIV's local UB slot; an explicit pop lane does not remap that
slot to the other physical AIV. Use the matching physical lane on this path.
The pinned GM-backed C2V implementation uses the lane for its GM band offset;
that route needs separate device validation.

PTO-ISA CPU simulation does not implement the explicit-lane overloads. Such
endpoints report a compile-time diagnostic under `__CPU_SIM`; the lane is
never removed to select a different overload. NPU execution and the PyTorch
reference FIFO are separate paths.

`tests/st/runtime/cross_core/test_explicit_aiv_lane.py` checks omitted,
static and dynamic lanes with exact goldens and twelve transfers through a
four-slot ring. Its A5 V2C cases swap the logical lanes to distinguish explicit
selection from an ignored operand. Local C2V cases use matching physical
lanes. A3 and A5 execution results must be recorded separately.

## A5 direct L1-to-UB Tile transfers

In a mixed core group, moving a Mat tile to Vec with
`pl.tile.move(tile, pl.Mem.Vec)` uses the A5 L1-to-UB DMA. The existing
`blayout` and `slayout` options select the destination layout. Request
`row_major` / `none_box` to unpack a standard NZ Mat tile into ND. In
`pl.split_aiv` regions, `pl.aiv_shard(mat_tile)` sends the corresponding
partition to each physical AIV. An ordinary Mat-to-Vec `tile.move` in a
dual-AIV mixed kernel is rejected because its physical lane is ambiguous.

The compiler reserves one shared receive buffer per AIV and selects two
intra-block event IDs alongside FIFO pipe reservations. Peer initializers can
specify the same even `flag_base` to reserve two event IDs for a unidirectional
pipe or four for a bidirectional pipe. PTOAS lowers
`pto.tdma_l1_to_ub` and `pto.tdma_ub_copy` with local synchronization for both
the event and buffer-ID modes. Use a PTOAS build that includes those ops.

Each transfer waits for both the remote DMA and the local copy to finish
before reusing the shared buffer. The local copy costs an extra UB read and
write and leaves an owned Tile that can outlive the next transfer. Both memory
planners can allocate the buffers. The default layout can preserve NZ, so
subsequent ND-only vector operations may insert layout conversions. Request
ND on the transfer for a consumer that expects ND.

This path accepts full, statically shaped, byte-addressable 2-D Tiles with
32-byte-aligned extents. It supports same-layout whole transfers and evenly
aligned ND/NZ partitions. The 12-bit burst count is split across commands
when necessary. Burst length and gaps must fit 16 bits, in 32-byte units.
PTOAS emits a counted loop over NZ column stripes.

::: pypto.language.system
    options:
      show_root_heading: false
      filters: ["!^_"]
