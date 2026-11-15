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

```python
lane = pl.cast(pl.tile.get_subblock_idx(), pl.INT64)
received = pl.system.tpop_from_aic(
    shape=[16, 16], dtype=pl.FP32, split=1, aiv_subblockid=lane
)
pl.system.tpush_to_aic(received, split=1, aiv_subblockid=lane)
pl.system.tfree_to_aic(received)
```

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

::: pypto.language.system
    options:
      show_root_heading: false
      filters: ["!^_"]
