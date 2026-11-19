# Async transfer validation

`pld.tensor.put_async` and `pld.tensor.get_async` lower on the A5 backend to a
dtype-suffixed external helper call (`pypto_urma_{put,get}_async_<dtype>`). The
helper validates the request against both endpoint tensor extents and the
communication window, requests the URMA workspace from the IR `engines` field,
and submits through the existing runtime entry point; it never polls.
Completion is owned by the enclosing task: submission registers the request's
completion condition, so a later task that consumes the destination as an input
is released only after the transfer retires. The remote endpoint (`put_async.dst`
or `get_async.src`) must be a window-bound `DistributedTensor`. The local endpoint
(`put_async.src` or `get_async.dst`) accepts a plain `Tensor` or a
`DistributedTensor` from another communication domain. Both endpoints must be
dense row-major. The caller supplies memory accessible to URMA; this API adds no
automatic MR registration or staging. The result preserves the type and storage
of `dst`; returning from the op does not imply completion.

Adapter symbols and C++ element types reuse the shared dtype conversions.

- Python builders require `engine=pld.CommEngine.URMA` (also the default).
  Integers, floats, booleans, strings, and unrelated enums are rejected before
  integer conversion. Raw IR stores the engine as integer `1`.
- `peer` must be an integer/index scalar. Negative constant ranks are rejected;
  this check does not establish runtime rank bounds for a communication domain.
- `dst_offsets`, `src_offsets`, and `shape` must be supplied together. A region
  must be contiguous on both endpoints: skip leading transfer dimensions known
  to equal one, then require every dimension after the first non-singleton to
  span its full extent. Dynamic extents may exceed one and are not skipped.
  Thus `[1, 32]` inside `[16, 64]` and `[1, 2, 64]` inside `[2, 4, 64]` are
  contiguous, but `[2, 32]` and `[2, 1, 64]`, respectively, are not.

## Supported async transfers

| Dimension | Current support | Not yet supported |
| --------- | --------------- | ----------------- |
| Ops | `pld.tensor.put_async` / `get_async` | native PTOAS async ops |
| Engine | URMA only (`pld.CommEngine.URMA`); SDMA/RDMA rejected by name | URMA / SDMA / RDMA selection |
| Target | A5 (Ascend 950) InCore/AIV body | additional targets once the helper ABI is ported |
| Payload | one contiguous region per call; the runtime splits transfers above 256 MiB internally | multi-SGE / scatter-gather lists |
| Completion | enclosing AIV task (normal done + registered async conditions); producer `InOut` → consumer `In` | explicit events, `wait`/`test`, per-request release |
| Registration | remote endpoint is window-bound; local endpoint may be a plain `Tensor` or belong to another domain | automatic MR registration / implicit staging |
| Source lifetime | caller must not overwrite `src` until the transfer task retires | runtime-managed lifetime |
| Remote visibility | receiver still needs the existing cross-rank notify/wait protocol | completion-ordered remote consumption |

**Usage pattern.** The transfer runs in the InCore body that owns the
destination. The destination is the producer's `InOut` output, and a later
task consumes it as `In`:

```python
data = pl.store(row, [my_rank, 0], data)
for peer in pl.range(nranks):
    if peer != my_rank:
        pld.tensor.put_async(
            data,
            peer=peer,
            src=data,
            dst_offsets=[my_rank, 0],
            src_offsets=[my_rank, 0],
            shape=[1, SIZE],
            engine=pld.CommEngine.URMA,
        )
return data
```

Returning from the op does **not** mean the bytes have moved. The consumer's
`In` dependency on `data` is the ordering point, and `src` must not be
overwritten until that task retires. Other compute in the producer task may
run before or after the submission, but it must not overwrite `src` while the
transfer is outstanding. A remote consumer still needs the
existing cross-rank notify/wait protocol: local task completion orders the
issuer, not the peer's reads.

**ABI note.** `CommContext` is 1104 bytes with four workspace engine slots
(SDMA, URMA, RDMA, CCU). The CCU slot extends the previous 1088-byte layout
without changing the rank or window offsets. The change is global, including
A2/A3 builds, so codegen and the runtime must move together. The async helper
reads the URMA workspace from `comm_workspace<CommWorkspaceKind::URMA>(*ctx)` and requires the domain
to request `workspace_engines=("urma",)` (the program-wide conservative
policy).

**Failure diagnostics.** Capacity is enforced transitively:
`allocation >= window view >= transfer region`. When a window is
materialized, its dense byte size (`product(shape) * dtype_bytes`) must fit the
allocation's logical size; an oversized view is rejected even if the program
only accesses a prefix. The physical cache-line padding of a slot is not
addressable through a view. The verifier also rejects a non-window remote endpoint, a
non-contiguous region, a missing offset/shape argument, a negative constant
peer, an invalid engine, and a transfer region that does not fit its endpoint
view. A symbolic view or slot size is checked on the host when the window declaration
executes, before its consumers. The generated helper rejects at runtime an invalid peer
(`peer < 0`, `peer >= nranks`, or `peer == rankId`), an out-of-range endpoint
or window byte range, a zero/unaligned byte count, and a missing workspace. All
failures are loud; there is no silent SDMA/RDMA fallback.

**Compatibility.** The return type (`dst`) and completion guarantee are stable.
Capacity validation adds no generated-kernel ABI change; symbolic view/slot
sizes are checked in host orchestration. Static window declarations are now
validated at compile time, so an existing program whose window view exceeds its
backing allocation is rejected instead of compiling. Support for sessions/events,
`wait`/`test`, SDMA/RDMA engines, native PTOAS lowering, per-request release and
multi-SGE will be added incrementally.
