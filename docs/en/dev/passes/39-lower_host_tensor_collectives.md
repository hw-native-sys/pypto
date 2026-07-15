# LowerHostTensorCollectives Pass

## Overview

`LowerHostTensorCollectives` rewrites host-orchestrator calls to
`pld.tensor.allreduce`, `pld.tensor.barrier`, `pld.tensor.broadcast`,
`pld.tensor.reduce_scatter`, `pld.tensor.allgather`, and
`pld.tensor.all_to_all` into compiler-internal
builtin chip dispatches. It runs
after [`MaterializeCommDomainScopes`](38-materialize_comm_domain_scopes.md), so
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
signal = pld.tensor.barrier(signal)
data = pld.tensor.broadcast(data, signal, root=0)
data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
data = pld.tensor.allgather(stage, data, signal)
data = pld.tensor.all_to_all(stage, data, signal)
```

For `allgather` / `all_to_all`, `stage` (TPUT source) and `data` (result)
must be two distinct windows. For `allgather` the `stage` window holds only
this rank's single chunk and is `[1, SIZE]`; for `all_to_all` it carries one
per-destination chunk per row and is `[NR, SIZE]`. In both cases `data` is the
`[NR, SIZE]` result window peers push into.
the pass emits the corresponding `builtin.tensor.*` dispatch per participating
device.  When the surrounding comm-domain scope has an explicit device list,
the pass emits a `SeqStmts`; otherwise it emits a sequential `for r in
pld.system.world_size()` loop.

Each generated builtin call carries the collective-specific args and kwarg
attributes from the source `pld.tensor.*` call.  Window-bound INOUT tensors
are threaded through as-is; scalar kwarg values (`op`, `root`, `dtype`) are
forwarded to the builtin.

Assignments preserve the user-facing rebind idiom by appending
`<result> = <original expr>` after the generated builtin calls.

## Checks

The pass requires both args to be materialized `DistributedTensorType` views in
the same `CommDomainScopeStmt`. The current host builtin path supports only
`ReduceOp.Sum` over FP32 data and an INT32 signal tensor shaped either as
rank-1 `[world_size]` or rank-2 `[world_size, 1]`, with enough static capacity
when the participating device count is statically known.

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
