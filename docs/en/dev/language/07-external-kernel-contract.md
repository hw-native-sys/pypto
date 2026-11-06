# External Kernel Contract (Internal v1)

This is the interface baseline for task 34.1 and [RFC #3293](https://github.com/hw-native-sys/pypto/issues/3293).
It defines data exchange and executable IR prototypes, not a public provider
registration API or an installed CANNBotDSL runtime adapter.

## Responsibilities

| Component | Owns | Must not own |
| --------- | ---- | ------------ |
| Provider | DSL recognition, compilation, metadata conversion | JIT AST state or runtime wrapper generation |
| Contract | Logical signature, effects, public returns, storage relationships | Live tensor addresses or mutable request state |
| JIT | Binding, specialization and effect/result normalization | Provider-specific branches in generic semantics |
| Runtime bridge | Native ABI conversion, target checks, task completion | Reinterpreting public returns as all written arguments |

The implementation lives in `python/pypto/jit/external/`; the bridge protocol is
in `python/pypto/backend/external_bridge.py`. Importing either does not import an
optional DSL. Registration/binding follows in 34.2, whole-program normalization
in 34.3, and native wrappers/runtime binding checks in 34.4.

## Requests and prepared kernels

`KernelRequest` carries ordered `ParameterSpec` records, compile-time constants,
a `CompileTarget` and effective options. `TensorSpec` records dtype, shape,
layout, strides, offset and alignment. Dimensions may be nonnegative integers
or request-local symbol names. Strides and offsets are in elements; an empty
stride tuple is unconstrained, not contiguous. `offset=None` means runtime-bound.
A scalar has an explicit `DataType`. Task IDs are not device scalar arguments.

Records are frozen dataclasses containing immutable tuples/frozensets. The
validators reject mutable nested inputs, duplicate names and non-finite constants.
Do not pass `SpecializeContext`, AST nodes, Python source expressions or actual
data pointers to a provider as normalized signature data.

`ExternalKernelProvider.prepare(kernel, request)` accepts an opaque provider-owned
reference and returns `PreparedKernel`:

| Field | Contract |
| ----- | -------- |
| `request` | Exact normalized input, including target/toolchain/options |
| `contract` | Logical parameters, effects, public returns, writebacks, alias permissions |
| `artifact` | Bundle-relative source/header paths and SHA-256 digests, entry, symbol, ordered include paths |
| `native_abi` | Independently versioned physical argument representation |
| `execution` | Core members, lanes, launch requirements and completion guarantee |
| `identity` | Provider/adapter versions, source/dependency digest and completeness promise |

The source bundle describes content identity, not storage ownership. Existing
artifact-store handles own materialized paths and lifetimes. Restore must verify
actual bytes and path/symlink containment; metadata validation alone does not.

## Public results and writebacks

Every tensor parameter has exactly one `TensorEffect`. Every written parameter
has exactly one `WritebackBinding`, regardless of whether the Python callable
returns it. `ReturnBinding` describes each public tensor return independently;
repeated and readonly parameter returns are legal. v1 returns alias the complete
argument view, not necessarily its complete backing allocation. Fresh allocations,
scalar device returns and arbitrary result-view expressions are not v1 results.

```python
from pypto.jit.external.contracts import (
    Access, Coverage, KernelContract, ParameterSpec, ReturnBinding,
    TensorEffect, TensorSpec, WritebackBinding,
)
from pypto.jit.external.normalization import plan_call
from pypto.pypto_core import DataType

parameters = tuple(
    ParameterSpec(name, TensorSpec((32,), DataType.FP32))
    for name in ("x", "state", "out")
)
contract = KernelContract(
    parameters,
    (
        TensorEffect(0, Access.READ),
        TensorEffect(1, Access.READ_WRITE, Coverage.PARTIAL),
        TensorEffect(2, Access.WRITE, Coverage.FULL),
    ),
    public_returns=(ReturnBinding(2),),
    writebacks=(WritebackBinding(1), WritebackBinding(2)),
)
plan = plan_call(contract)
assert plan.result_parameters == (1, 2)
assert plan.public_slots == (1,)
assert plan.writeback_slots == ((1, 0), (2, 1))
```

For a public `result = stage(x, state, out)`, the internal call returns updated
`state` and `out`; only the latter is projected as `result`. Task 34.3 must also
replace subsequent state uses and propagate the effect through inline boundaries.
The plan itself does not rewrite Python source. Materialize its projections as
IR before passes run: slot numbers are not stable across `NormalizeReturnOrder`.

IR prototypes use existing `external_return_params` with references to the
function's own parameters, together with `param_directions` and return types.
This survives parameter reconstruction and avoids guessing aliases by shape or
output position. No new IR node or binding is needed for this representation.
See [existing extern return declarations](04-external-kernels.md).

## Partial writes, views and aliases

| Case | v1 rule |
| ---- | ------- |
| Read or no access | Conservatively `In`; no dependency elimination for unused parameters |
| Write-only, full argument coverage | `Out`; ordinary call-direction derivation still applies |
| Read/write, partial or unknown coverage | `InOut`, preserving incoming dependencies |
| Partial/unknown write without untouched-byte preservation | Reject |
| Full write to a view | Does not imply full write to the root allocation |
| Distinct arguments overlap and either is written | Require explicit `AliasPermission` and bridge support |
| Unproven overlap without a permission | Prove/check disjointness before invocation, otherwise reject |

Missing/unknown access is rejected; unknown *coverage* of a known write may be
conservative. Alias permissions guarantee kernel-internal safety, not merely
scheduling order. Runtime alias/alignment checks apply on cache hits too.

Do not reinterpret `ReturnedParamIndices == None` as proof of fresh storage.
Full-parameter return lineage deliberately stops at a changed view; storage-region
analysis has a different purpose. Preserve view shape, strides and offset. Zero-trip
loops and untaken branches retain the incoming storage and dependencies.

## Native ABI and execution

`NativeABI` names a layout and version understood by a bridge. Its parameter tuple
is in **physical order**. Each `ABIParameter` names a logical parameter, value kind,
byte size, byte alignment, optional tensor axis, and element/byte unit. Scalar bits
are copied without numeric conversion. Descriptor internals belong to the named
ABI/version; validating a name is not proof that a bridge supports that layout.
Data pointers denote allocation bases; apply view offsets exactly once.

`ExecutionRequirements` declares AIC/AIV membership and AIV lane count. Standalone
AIV uses one lane; mixed tasks may use one or two. Full residency requires a
synchronized start; full-resident mixed tasks require both AIV lanes. Logical
launch context is explicit so wrappers do not substitute physical core numbers.

The only v1 completion guarantee is `ENTRY_RETURN`: all work has finished and
writes are visible at entry return. A bridge must supply the required waits/cache
operations before runtime task completion. Unsupported asynchronous completion
must fail, never silently degrade to a plain call.

`validate_target` checks the exact compile target, supported ABI name/version and
semantic/execution capabilities. Requirements derived from actual effects and
launch fields cannot be hidden by omitting a capability label. It does not prove
per-invocation aliasing, dimensions, alignment or occupancy; the bridge must do so.

## Versions and cache identity

Unknown schema versions and required capabilities fail with `ContractError`.
Native ABI versions are separate from the schema and host runtime ABI. Add new
semantics with explicit capability checks; do not infer them from provider names.

`preparation_key` covers normalized signature, constants/options, target/toolchain,
provider/adapter versions and the source/dependency digest. Named constants/options
are order-independent; physical argument order and include search order are not.
Incomplete dependency identity returns `None`: prepare again. `prepared_digest`
fingerprints validated output, including effects, ABI and execution requirements.
A downstream executable key must additionally cover bridge version, runtime and
effective build options. Reuse `ArtifactStore` for locking, publication and leases.

## Validation evidence and limits

`test_external_contract.py` covers two independent fake providers, result/writeback
planning, immutable requests, diagnostics, ABI order/units, target capability gates
and identity invalidation. `test_external_ir_semantics.py` checks normalized forms
through round-trip serialization, the default pass pipeline and orchestration
code generation: explicit public return plus hidden state writeback, inline,
zero/multi-trip loops, branches, AIC/AIV/MIX and a view-to-root dependency.

These are CPU compiler checks. They do not establish numerical device correctness,
automatic source-level writeback propagation, safe overlapping execution, or full
runtime completion behavior. Those require the subsequent integration tasks and
focused device tests. Existing extern APIs continue to use their compatibility path.
