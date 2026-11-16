# Intra-Block Synchronization

## Event forms and pipes

These operations synchronize AIC (Cube) and AIV (Vector) within the same execution block.
`pl.system.set_intra_block` signals an event; `pl.system.wait_intra_block` waits for
its corresponding `event_id`. They emit PTOAS `pto.set_intra_block` and
`pto.wait_intra_block`, respectively, and return no value. The existing `sync_set`
FFTS modes 0/1/2 contract is unchanged. On A5 the static ID range is **0–31**, and a dynamic ID
must be `pl.Scalar[pl.INT32]` or `pl.Scalar[pl.INT64]`. Bare DSL integer
literals remain static attributes. Cast a dynamic INDEX scalar explicitly; no implicit conversion or dynamic range check is inserted.
Legal A5 pipes are `FIX`, `MTE1`, `MTE2`, `MTE3`, `V`, and `S`; `M` and `ALL`
are rejected. `S` requires a PTOAS build whose A5 intra-block event verifier accepts
it; older assemblers can reject the emitted PTO. Set and wait may use different legal pipes: each pipe orders
its local producer or consumer, while the event identifies the rendezvous.
A2/A3 PTOAS uses FFTS mode 2, restricts IDs to 0–15, and rejects `S`; do not
assume the A5 range or per-AIV semantics on those targets.

```python
# Placement example only, not a complete handshake.
# The wait runs on AIV1; select that lane in the surrounding kernel.
pl.system.set_intra_block(17, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
pl.system.wait_intra_block(1, pipe=pl.PipeType.V, core_type=pl.KernelType.AIV)
```

## Pipeline ordering

A `pl.pipeline` body containing an intra-block event, even inside nested control,
is conservatively lowered to a sequential loop for every `stage` value.
This preserves the authored wait → payload access → set order and event
reuse across iterations. Intra-block events have no SSA dependency on those accesses;
IO clustering and software pipelining must not move them across the handshake.
No additional synchronization is inserted.

## Kernel placement and AIV banks

`core_type` selects AIC versus AIV, **not AIV0 versus AIV1**. Use it in mixed
kernels; expansion rejects an unpinned intra-block event rather than duplicating
its side effect. In explicitly typed AIC/AIV functions it can be omitted.
There is no Python lane parameter. The installed A5 PTO-ISA `TPush.hpp` uses
16 local event IDs and `VEC_CORE_ID_OFFSET = 16`: Cube addresses AIV0 with
`e` and AIV1 with `e + 16`, for `e` in 0–15; each AIV uses its local `e`.
Cube must signal/wait separately for each participating AIV. A raw event 17
on Cube therefore addresses AIV1's local event 1, not local event 17 on both
AIVs. Executing on only one AIV also requires lane control in the kernel.
The single-AIV runtime protocol uses a Cube MTE1 wait followed by a
`PIPE_ALL` barrier before reusing the event, so scalar dispatch cannot outrun
the acknowledgement. Intra-block `PIPE_S` assembler support is checked separately.

## Resource ownership

These calls do not reserve IDs. TPipe uses the same
intra-block namespace: a unidirectional pipe consumes `flag_base` and
`flag_base + 1`, and a bidirectional pipe also consumes `+2` and `+3`, with
Cube-side AIV1 aliases at `+16`. PTOAS's pipe allocator accounts for pipe
components, not manual intra-block events. Avoid every live pipe interval and its
aliases, and the events used by mixed `SYNCALL` (consult the target
PTO-ISA constants: the inspected A5 headers use local 12/13 and
Cube-side aliases 28/29). The FFTS API's 14/15 reservation does not make those
numbers free for intra-block events. Automatic intra-core `set_flag`/`wait_flag`
uses a separate namespace. The protocol author must allocate non-conflicting
IDs, balance every set/wait across iterations, and provide appropriate
payload visibility and buffer-reuse ordering.

## Native payload validation

The reused-buffer acceptance cases publish Cube Acc tiles to GM with FIX,
wait on AIV MTE2, and acknowledge on AIV MTE3 before the next Cube overwrite.
They validate ready/ack ordering with exact iteration-tagged payloads for each
AIV lane. An explicit `CORE_GROUP` co-schedules the Cube producer and the
per-lane AIV consumer; the manual GM protocol stays separate from automatic
direct C2V promotion in mixed kernels.
That promotion rejects GM exchanges combined with authored intra-block events,
just as it does for authored FFTS events; use explicit peers for manual protocols.
This DMA-based coverage does not establish scalar D-cache to DMA
publication semantics.

## Debug backend support

Torch debug code generation rejects programs containing these operations:
its mixed-thread runtime does not model intra-block event ordering or AIV event banks.
Use the native PTO backend for such programs. Silently omitting a rendezvous
could otherwise let a consumer read a shared payload before its producer.
