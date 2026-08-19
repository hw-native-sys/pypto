# MaterializeDistTensorCtx Pass

Materializes one explicit `CommCtxType` parameter and argument for each
`DistributedTensorType` function parameter.

## Overview

Distributed tensors need a communication context at every dispatch boundary:
host orchestration passes a per-rank `device_ctx`, L2 orchestration forwards it
through task args, and L1 PTO codegen uses it to lower `pld.system.rank`,
`pld.system.nranks`, `notify`, `wait`, `put`, and remote memory ops.

Older codegen paths synthesized those ctx values independently at several
sites. This pass makes the ctx flow explicit in IR instead:

1. For every function with `DistributedTensorType` parameters, append matching
   `CommCtxType` parameters at the tail of the signature, in distributed-tensor
   parameter order. The appended parameters are `ParamDirection::In`.
2. For every `Call` / `Submit` to such a function, append matching ctx args.
   If the distributed tensor arg is a caller parameter or an SSA alias of one,
   forward the caller's materialized ctx parameter. Return positions are
   matched to the callee's returned parameters, so mixed or reordered return
   tuples do not fall back to positional tail alignment. Tensor aliases carried
   through `ForStmt` / `WhileStmt` are tracked as well. In host orchestration
   only, if the lineage cannot be resolved, bind
   `pld.system.get_comm_ctx(dist)` immediately before the call and pass that
   result. Chip orchestration and device functions must resolve an explicit
   context; an unresolved argument is diagnosed instead of synthesizing a
   device-side query.
3. In chip orchestration and device functions, replace every
   `pld.system.get_comm_ctx(dist)` with the resolved explicit `CommCtxType` SSA
   value. Host orchestration keeps the op because host codegen resolves it from
   the window's per-rank runtime context.
4. If call-site `arg_directions` are already resolved, append matching
   `ArgDirection::Scalar` entries so downstream codegen can keep treating ctx as
   ordinary scalar task payload.

This pass does not add `CommCtxType` values to `IfStmt` return variables or
branch yields. DistributedTensor `if` lowering keeps the existing requirement
that both branches refer to the same backing/context; dynamic context merges
remain outside this change (issue #2027).

The pass runs after `LowerHostTensorCollectives` and before the final
`Simplify`. At that point host window buffers have already been materialized by
`MaterializeCommDomainScopes`, host tensor collectives have been lowered, and
there is still time for the final simplify pass to clean up any forwarding
aliases.

## Why CommCtx Is Different From Dynamic Dims

Dynamic tensor dimensions can be recovered locally from tensor descriptors at
the wrapper boundary. A communication context cannot: it is real dataflow across
host -> orchestration -> task payload -> kernel signature. Keeping it in IR
prevents codegen sites from drifting out of sync.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MaterializeDistTensorCtx()` | `passes.materialize_dist_tensor_ctx()` | Program-level |

```python
from pypto.pypto_core import passes

program = passes.materialize_dist_tensor_ctx()(program)
```

## Example

Before:

```python
def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
    return self.kernel(data)

def host_orch(self):
    data = pld.window(buf, [256], dtype=pl.FP32)
    self.chip_orch(data, device=r)
```

After:

```python
def chip_orch(self, data, data_ctx: pld.CommCtxType):
    return self.kernel(data, data_ctx)

def host_orch(self):
    data = pld.window(buf, [256], dtype=pl.FP32)
    data_ctx = pld.system.get_comm_ctx(data)
    self.chip_orch(data, data_ctx, device=r)
```

The kernel body does not need to change. Existing
`pld.system.get_comm_ctx(data)` uses in a device function are rewritten to the
explicit ctx parameter by this pass; host-orchestration uses remain runtime
queries.
