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
   tuples do not fall back to positional tail alignment. This holds for `Submit`
   too: its result positions are the callee's return positions (the trailing
   `Scalar[TASK_ID]` has no ctx). Builtin ops that bind a fresh SSA var to a
   DistributedTensor that already exists forward that value's ctx — both
   output-side writebacks, which the op declares itself via
   `set_output_reuses_input(idx)` (`tile.store` -> `args[2]`,
   `tensor.assemble` -> `args[0]`), and zero-copy buffer-aliasing views
   (`tensor.view`, `tile.slice`, `tensor.reshape`, ...), whose result type
   propagates `DistributedTensorType::window_buffer_` from `args[0]`.
   Tensor aliases carried through `ForStmt` / `WhileStmt` are tracked as well.
   In host orchestration only, if the lineage cannot be resolved, bind
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

A loop carry is subject to the same one-context rule. The carry is seeded from
its init value before the body is walked, so a self-carry
(`data = self.comm(data)`) resolves; the value yielded back into the carry is
then checked against that seed, and rebinding the carry to a *different*
DistributedTensor inside the loop is diagnosed rather than silently taking the
init value's context. A yield whose lineage this pass cannot trace at all leaves
the seed in place.

The pass produces `IRProperty::DistTensorCtxMaterialized`: no
`pld.system.get_comm_ctx` survives outside host orchestration. The pass enforces
this for every function it rewrites, and the property verifier checks it
independently — which also covers Programs the pass returns untouched because no
function declares a `DistributedTensorType` parameter. The property is listed in
`GetVerifiedProperties()`, so the pipeline checks it on the default verification
level rather than only when a test installs a `VerificationInstrument`.

It requires `IRProperty::ReturnParamsExplicit`: the return-position map comes
from `return_lineage::ExplicitReturnedParamIndices`, a pointer-identity read of
the `ReturnStmt` that is only meaningful once `NormalizeReturnOrder` has
canonicalized it.

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
