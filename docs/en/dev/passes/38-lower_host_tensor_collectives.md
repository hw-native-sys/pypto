# LowerHostTensorCollectives Pass

## Overview

`LowerHostTensorCollectives` rewrites host-orchestrator calls to
`pld.tensor.allreduce` into compiler-internal builtin chip dispatches. It runs
after [`MaterializeCommDomainScopes`](37-materialize_comm_domain_scopes.md), so
each window-bound data tensor and explicit signal tensor already has a
`WindowBuffer` back-reference and belongs to an inferred communication domain.

The pass does not change non-host functions. InCore allreduce calls continue to
use [`LowerCompositeOps`](14-lower_composite_ops.md).

## Position in the pipeline

```text
... -> MaterializeCommDomainScopes -> LowerHostTensorCollectives -> Simplify (final) -> MaterializeRuntimeScopes
```

The final `Simplify` runs after this pass so any generated loop bounds or
constant expressions can still be folded before runtime scopes are inserted.

## Behavior

For a host-orchestrator call:

```python
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
```

the pass emits one `builtin.tensor.allreduce` call per participating device.
When the surrounding comm-domain scope has an explicit device list, the pass
emits a `SeqStmts`; otherwise it emits a sequential `for r in
pld.system.world_size()` loop.

Each generated builtin call:

- uses the same `data` and `signal` args,
- carries `attrs["device"]`, `attrs["op"]`, and `attrs["dtype"]`,
- marks both args `InOut`,
- returns the same distributed tensor type as `data`.

Assignments preserve the user-facing rebind idiom by appending
`data = <original data expr>` after the generated builtin calls.

## Checks

The pass requires both args to be materialized `DistributedTensorType` views in
the same `CommDomainScopeStmt`. The current host builtin path supports only
`ReduceOp.Sum` over FP32 data and a rank-1 INT32 signal tensor with enough
static capacity when the participating device count is statically known.

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
