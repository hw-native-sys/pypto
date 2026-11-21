# Manual Buffer-ID Synchronization (A5)

Part of the [Operator System](05-operators.md) reference.

This is the manual Buffer-ID subset of [#3268](https://github.com/hw-native-sys/pypto/issues/3268).

`pl.system.get_buf(buf_id, *, pipe, mode=0)` and
`pl.system.rls_buf(buf_id, *, pipe, mode=0)` order accesses to a local buffer
between pipes on the **same core**. Both are side-effecting statements with no
result (`UnknownType` in PyPTO IR). Supported concrete pipes are `MTE1`, `MTE2`,
`MTE3`, `M`, `V`, `S`, and `FIX`; `ALL` is rejected. Use only pipes available
on the enclosing core. A2/A3 targets and every nonzero mode are rejected.

A Python integer ID is a static attribute in `[0, 31]` (booleans are rejected).
An INDEX expression, including an INDEX constant, is a dynamic SSA operand;
other scalar dtypes are rejected. The kernel author must keep dynamic IDs in
`[0, 31]` at runtime, pair get/rls on every path, and protect **all** accesses
that share a buffer. No runtime ID checks or protocol verification are inserted.

```python
# Interface fragment only: surround the actual accesses with matching calls.
pl.system.get_buf(0, pipe=pl.PipeType.MTE2)
# Load into the protected local buffer here.
pl.system.rls_buf(0, pipe=pl.PipeType.MTE2)
pl.system.get_buf(buf_id, pipe=pl.PipeType.V, mode=0)  # buf_id: Scalar[INDEX]
# Consume the buffer here, then release it on the same pipe and ID.
pl.system.rls_buf(buf_id, pipe=pl.PipeType.V)
```

Use explicitly typed `FunctionType.AIC` / `FunctionType.AIV` kernels. InCore
and mixed-kernel placement is rejected before expansion because a pipe such
as MTE2 does not identify the owning core. The default pipeline retains the
calls inside loops and keeps automatic event synchronization enabled. T1 adds
no automatic synchronization strategy selector. PTOAS BufidSync skips a whole
function containing any manual static **or dynamic** Buffer-ID operation;
it does not merge manual and automatic Buffer-ID allocation. These APIs do
not provide cross-core mode-4 synchronization or the full hardware Mutex API.

Auto-tiling also preserves these boundaries. A delayed M/N matmul/store fold
crossing a Buffer-ID operation is rejected when the complete result cannot fit
in L0C. Place the store inside the protected region or tile the computation
explicitly; a smaller result can use a K-only plan at its original position.

These APIs require the backend fixes in [PTOAS !462](https://gitcode.com/cann/pto-as/merge_requests/462).
The general minimum PTOAS version does not establish that capability. When a
generated kernel contains manual Buffer-ID operations, PyPTO probes the selected
assembler for concrete-pipe EmitC lowering and dynamic manual-sync BufidSync
skipping, and reports a dependency error if either is missing. Set `PTOAS_ROOT`
to a build containing those fixes. Ordinary kernels retain their baseline
version requirement; successful feature probes are cached per executable.

CPU simulation is outside the validated support of this change and is skipped
by the task's acceptance policy. The device acceptance cases in
`tests/st/runtime/ops/test_buffer_sync.py` use three pipes, static reuse and
dynamic physical ping-pong slots, with independent golden data and repeated
launches. They disable automatic event insertion only in the test harness;
the public default compilation policy remains unchanged.
