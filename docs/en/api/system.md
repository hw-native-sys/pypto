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

In FIFO mode, explicit lanes require an AIV-side Vec Tile entry and split
codes 1/2/3/4. Existing split, shape, layout and initialization contracts
still apply, including the restriction on odd V2C push codes 3/4.

On A5, `pl.system.tpush_to_aiv(acc, split=0, dst=mailbox, aiv_subblockid=1)`
selects AIV1 for a direct FixPipe writeback from AIC. This form takes a
compile-time Python integer 0 or 1 and converts the entire FP32 Acc into
a same-shaped contiguous BF16 Vec mailbox at offset `[0, 0]`. The row pitch
must be 32-byte aligned. For a mailbox slice, the compiler must also be able
to prove 32-byte alignment of its base address. The caller supplies the peer
UB address and orders the producer and consumer with explicit synchronization.
The matrix describes compiler contracts; device execution needs separate
validation for each architecture.

| Path | A3 | A5 |
| ---- | -- | -- |
| Explicit lane, Vec push/pop, split 1/2 | PTOAS operand supported | PTOAS operand supported; V2C layout adapter retained |
| Explicit lane, C2V pop, split 3/4 | Existing GM-backed odd split rules apply | Existing GM-backed odd split rules apply |
| Explicit lane, V2C push, split 3/4 | PyPTO restriction retained | Backend rejects odd V2C |
| FIFO lane with split=0, GlobalTensor, or on the AIC side | Rejected | Rejected |
| Direct `tpush_to_aiv` with BF16 `dst`, split=0, and static lane | Rejected | FixPipe writeback to the selected physical AIV |

In FIFO mode the author must keep dynamic lanes in `{0, 1}`, provide each logical
lane exactly as required by the matching split protocol, and pair push/pop/free
counts and pipe IDs. No runtime range guard is inserted. A constant outside
that range is rejected. Selecting a logical lane does not move the operation
to another physical core or change the executing kernel.

The FIFO lane follows the backend transport. A5's Acc→UB C2V pipe returns
the executing AIV's local UB slot; an explicit pop lane does not remap that
slot to the other physical AIV. Use the matching physical lane on this path.
The pinned GM-backed C2V implementation uses the lane for its GM band offset;
that route needs separate device validation.

PTO-ISA CPU simulation does not implement the explicit-lane FIFO overloads. Such
endpoints report a compile-time diagnostic under `__CPU_SIM`; the lane is
never removed to select a different overload. NPU execution and the PyTorch
reference FIFO are separate paths.

`tests/st/runtime/cross_core/test_explicit_aiv_lane.py` checks omitted,
static and dynamic lanes with exact goldens and twelve transfers through a
four-slot ring. Its A5 V2C cases swap the logical lanes to distinguish explicit
selection from an ignored operand. Local C2V cases use matching physical
lanes. A3 and A5 execution results must be recorded separately.

## L1 to UB boundary

Explicit lane selection does not establish a Mat (L1) to Vec (UB) transfer.
The current PTOAS `verifyTMovGenericPairing` has no Mat→Vec pair, and the 950
memory graph in `src/backend/common/soc.cpp` has no such edge. Keep this
restriction until a legal backend mechanism is demonstrated.

The alternatives require separate contracts:

- Direct TMOV needs a legal Mat→Vec instruction, layouts, dtypes, valid extents
  and producer/consumer ordering. The minimal assembler rejection is covered
  by `tests/ut/codegen/test_pto_codegen_aiv_lane.py`.
- A tile TPipe needs evidence that its C2V producer accepts Mat storage and
  that its AIV consumer receives Vec storage; the existing Acc→Vec pipe is
  not that evidence. A Mat producer probe currently fails on both A3 and A5
  with `tile type must map to a supported producer pipe`.
- A GM route needs a legal L1→GM store, AIV GM→UB load, lifetime, cache
  visibility and synchronization. A3's store verifier permits Mat, but A5's
  permits only Vec or Acc. Reloading the original GM input on AIV can be a
  separately authored algorithm when that input is still available; it does
  not copy an arbitrary existing L1 tile.

No new memory-graph edge or implicit staging is introduced. These compiler
limits do not establish a hardware capability limit. Backend instruction and
protocol evidence, including the intended meaning of the external L1→UB
request, remains necessary before extending this path.

::: pypto.language.system
    options:
      show_root_heading: false
      filters: ["!^_"]
