# LowerHostTensorCollectives Pass

## Overview

`LowerHostTensorCollectives` rewrites host-orchestrator calls to
`pld.tensor.allreduce`, `pld.tensor.barrier`, `pld.tensor.broadcast`,
`pld.tensor.reduce_scatter`, `pld.tensor.allgather`,
`pld.tensor.all_to_all`, and `pld.tensor.all_to_all_v` into compiler-internal
builtin chip dispatches. It runs
after [`MaterializeCommDomainScopes`](42-materialize_comm_domain_scopes.md), so
each window-bound data tensor and explicit or synthesized signal tensor already has a
`WindowBuffer` back-reference and belongs to an inferred communication domain.

The pass does not change non-host functions. InCore allreduce calls continue to
use [`LowerCompositeOps`](12-lower_composite_ops.md).

## Position in the pipeline

```text
... -> SynthesizeAllReduceSignals -> MaterializeCommDomainScopes -> LowerHostTensorCollectives -> MaterializeDistTensorCtx -> Simplify (final) -> MaterializeRuntimeScopes
```

The final `Simplify` runs after this pass so any generated loop bounds or
constant expressions can still be folded before runtime scopes are inserted.

## Behavior

For a host-orchestrator call:

```python
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, core_num=4)
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
signal = pld.tensor.barrier(signal)
data = pld.tensor.broadcast(data, signal, root=0)
data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
data = pld.tensor.allgather(stage, data, signal)
data = pld.tensor.all_to_all(stage, data, signal)
data = pld.tensor.all_to_all_v(input, target, signal, send_counts, recv_counts)
```

`pld.tensor.allreduce` dispatches on its `mode` kwarg: the default
`mode="mesh"` lowers to `builtin.tensor.allreduce`, while `mode="ring"` lowers
to `builtin.tensor.allreduce_ring`. Any other value is rejected as a user
error.

For `allgather` / `all_to_all` / `all_to_all_v`, `stage`/`input` (TPUT source)
and `data`/`target` (result) must be two distinct windows. For `allgather` the
`stage` window holds only this rank's single chunk and is `[1, SIZE]`; for
`all_to_all` it carries one per-destination chunk per row and is
`[NR, SIZE]`; for `all_to_all_v` it carries one `MAX_RECV`-row capacity block
per destination and is `[NR*MAX_RECV, SIZE]`. In both `all_to_all` /
`all_to_all_v` cases `data`/`target` is the peers'-push-in result window.
`all_to_all_v` additionally requires `send_counts` (window-bound at this
layer, LOCAL-only) and `recv_counts` (window-bound, published cross-rank via
`pld.system.notify`) — all five window args must resolve into the same
`CommDomainScopeStmt` and must be pairwise-distinct window allocations
(aliasing any pair is a cross-process race, whether data-vs-data,
data-vs-control, or control-vs-control).

The pass emits the corresponding `builtin.tensor.*` dispatch per participating
device (including `builtin.tensor.allreduce` /
`builtin.tensor.allreduce_ring`, `builtin.tensor.barrier`,
`builtin.tensor.broadcast`, `builtin.tensor.reduce_scatter`,
`builtin.tensor.allgather`, `builtin.tensor.all_to_all`, and
`builtin.tensor.all_to_all_v`). When the surrounding comm-domain scope has an
explicit device list, the pass emits a `SeqStmts`; otherwise it emits a
sequential `for r in
pld.system.world_size()` loop.

Each generated builtin call carries the collective-specific args and kwarg
attributes from the source `pld.tensor.*` call.  Window-bound INOUT tensors
are threaded through as-is; scalar kwarg values (`op`, `root`, `dtype`, and
mesh-AllReduce `core_num`) are forwarded to the builtin. `all_to_all_v`'s
`MAX_RECV` is not a lowering-time attribute: the HOST kernel derives it at
entry as `target.shape[0] / nranks` (the runtime comm-domain size), so no
per-`MAX_RECV` codegen variant mangling is needed and the block layout stays
consistent with the devices actually running.

Assignments preserve the user-facing rebind idiom by appending
`<result> = <original expr>` after the generated builtin calls.

## Printed form

The `builtin.tensor.*` operators are `internal_only` in the registry: no DSL
wrapper spells them, and the user-facing op-creation path rejects them by name.
The python printer still has to render them, and does so under the `pl.builtin`
namespace — the same `pl.` prefix it puts on any non-`pld` registered operator:

```python
for r_1 in pl.range(pl.const(0, pl.INT64), pld.system.world_size(), pl.const(1, pl.INT64)):
    pl.builtin.tensor.allreduce(
        data, signal, op=0, dtype=pl.FP32, core_num=1,
        attrs={"op": 0, "dtype": pl.FP32, "core_num": 1, "device": r_1,
               "arg_directions": [pl.adir.inout, pl.adir.inout]},
    )
```

The parser reads that spelling back (`ast_parser._parse_builtin_op`), so the
lowered dispatch survives print -> parse. It is a machine-only surface, scoped
to names actually registered under `builtin.`, and it accepts only what the
printer can write: the `device` and `arg_directions` attrs are required, since
orchestration codegen reads both back behind internal checks. A hand-written
call omitting them is rejected as a user error at parse time rather than
surfacing as a compiler-bug diagnostic during codegen. Write the composite
`pld.tensor.*` form instead.

Note that a whole-program `assert_structural_equal` round-trip is still blocked
one pass upstream: [`MaterializeCommDomainScopes`](42-materialize_comm_domain_scopes.md)
synthesizes `CommDomainScopeStmt` (printed as a leading comment) and the
`WindowBuffer` back-references on `DistributedTensorType` (not printed), and
neither has a DSL surface to parse back.

## Checks

The pass requires both args to be materialized `DistributedTensorType` views in
the same `CommDomainScopeStmt`. The host allreduce builtin supports
`ReduceOp.Sum`, `Max`, `Min`, and `Prod` over FP16 or FP32 data and arbitrary
positive element counts. It processes 256-element chunks and rounds ragged FP16
and FP32 load spans to 32 bytes without changing the logical tensor shape.
Its INT32 signal tensor may be rank-1 `[world_size]` or rank-2
`[world_size, signal_stride]`, with enough static capacity when the
participating device count is statically known. Because the signal is produced
by `pld.window`, it is packed by construction and the builtin indexes it as a
flat row-major array.

Mesh allreduce takes one signal lane per launched AIV block: a rank-1 signal is
valid only when `core_num == 1`, and a rank-2 signal needs a constant
`signal_stride >= core_num` (a wider stride is accepted, so an explicit signal
may carry spare lanes). `core_num` must also fit the configured backend's AIV
core count — the builtin is submitted as a standalone AIV kernel with
`require_sync_start`, so an over-subscribed launch could never be admitted and
would hang instead of failing. The bound is skipped when no backend is
configured (pure-IR tests). Multicore is mesh-only: `mode="ring"` requires
`core_num == 1`.

Ring allreduce (`mode="ring"`) uses a rank-2 signal shaped
`[2 * (NR - 1) + 1, NR]`, whose `shape[0]` must equal `2 * (NR - 1) + 1` when both
signal dimensions are compile-time constants, and must be at least
`2 * (NR - 1) + 1` when only `shape[0]` is statically known (no static check when
both dims are dynamic). When the participating device count is statically known, the signal
must have enough static capacity. Ring allreduce additionally requires `numel(src) % NR == 0`
(the ring schedule partitions src into NR contiguous chunks; a non-zero remainder would leave a
trailing partial chunk the kernel cannot handle). The host-ring `src` shape must be
statically known — dynamic extents are rejected, since the kernel would otherwise silently
return unreduced data when the runtime `numel` is not divisible by `NR`.

Ring allreduce currently supports only `ReduceOp.Sum` with `dtype=FP32`.
`ReduceOp.Max`, `ReduceOp.Min`, `ReduceOp.Prod`, and `FP16` are not yet available
with `mode="ring"`. Ring allreduce also supports at most 16 participating
devices (`world_size <= 16`).

All window operands of a HOST collective — data and signal alike — must
resolve to pairwise distinct `WindowBuffer` allocations. Two `pld.window()`
views over the same `alloc_window_buffer` are a cross-process data race under
in-kernel TPUT/notify: data-vs-data is a reduce overwrite, data-vs-control
races a notify/count write against a kernel read, and control-vs-control races
a notify against a count publish. `LowerHostTensorCollectives` rejects any
aliasing pair before emitting the builtin dispatch.

`broadcast`'s `root` kwarg is additionally bounds-checked when the
participating device count is statically known: on an explicit static device
subset it must satisfy `root < participating device count`. The fully-dynamic
"all device" domain cannot be checked at compile time (no device count is
known there) — the same documented limitation as the signal-capacity check.

`all_to_all_v`'s single-use Set(1)/wait≥1 signal cannot be reused across a
`for`/`while` loop in `host_orch` — [`MaterializeCommDomainScopes`](42-materialize_comm_domain_scopes.md),
which runs immediately before this pass, rejects that case up front (the same
restriction `LowerCompositeOps` enforces on the InCore path). On an explicit
static device subset, `all_to_all_v`'s signal `shape[0]` must exactly equal
the subset size (not merely `>=`, as required for the other collectives),
since `MAX_RECV` is derived as `target.shape[0] / signal.shape[0]` and an
over-provisioned signal would silently mis-lower.

## Pass properties

| Field | Value |
| ----- | ----- |
| `required` | `{IRProperty::CommDomainScopesMaterialized}` |
| `produced` | `{IRProperty::CommDomainScopesMaterialized}` |
| `invalidated` | `{}` |

## Reference

- Source: [src/ir/transforms/lower_host_tensor_collectives_pass.cpp](../../../../src/ir/transforms/lower_host_tensor_collectives_pass.cpp)
- Header: [include/pypto/ir/transforms/passes.h](../../../../include/pypto/ir/transforms/passes.h)
- Tests: [tests/ut/ir/transforms/test_lower_host_tensor_collectives.py](../../../../tests/ut/ir/transforms/test_lower_host_tensor_collectives.py)
