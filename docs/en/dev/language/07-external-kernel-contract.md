# External Kernel Contract (Internal v1)

This is the interface baseline for tasks 34.1/34.2 and [RFC #3293](https://github.com/hw-native-sys/pypto/issues/3293).
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
optional DSL. Internal registration and capture are implemented in `registry.py` and
`compatibility.py`. Whole-program normalization follows in 34.3, and native
wrappers/runtime binding checks in 34.4.

## Provider discovery and capture

`ProviderRegistry` is explicitly scoped. Registration separates a cheap
`matches(object) -> bool` predicate from a `load() -> CapturingKernelProvider`
factory. Neither registration nor discovery loads a compiler, captures inputs,
or compiles kernels. Matchers must only recognize objects without importing an
optional DSL. No implicit plugin scanning or global kernel cache is installed.

```python
from pypto.jit.external.registry import ProviderRegistry

# ExampleKernel and ExampleProvider are supplied by the integration.
registry = ProviderRegistry()
registry.register("example", matches=lambda obj: isinstance(obj, ExampleKernel),
                  load=ExampleProvider)
ref = registry.discover(kernel)  # None when no provider recognizes the object.
# Explicit selection is available when multiple providers recognize an object.
ref = registry.bind(kernel, provider="example")
captured = ref.capture(request)  # Validated signature and target are now known.
prepared = captured.prepare()
```

Multiple matches are errors; registration order never chooses a winner. Duplicate
names and non-boolean matcher results also fail. Explicit binding selects a named
provider without running matchers. A `KernelRef` strongly owns its kernel and
registration; the registry retains neither discovered kernels nor captures.
Removing a registration affects future discovery, while existing refs remain usable.
Provider factories load once per registration, including concurrent captures.
Providers must support independent concurrent requests or serialize their own compiler.

A capturing provider implements `capture(kernel, request) -> CapturedInputs` and
`prepare(captured, request) -> PreparedKernel`. `CapturedInputs` owns immutable
provider-encoded bytes and a `PreparationIdentity`. Its digest must match those
bytes. Providers encode all relevant source, referenced constants and dependencies
into the snapshot, and decode that snapshot during preparation. The `complete`
flag is still a provider promise of completeness; digest verification cannot prove
that an undeclared dependency was included. `False` yields no reusable preparation
key. `CapturedKernel.prepare()` always prepares afresh; task 34.6 adds artifact-store
reuse. Output validation rejects a changed request or capture identity.
The earlier prepare-only `ExternalKernelProvider` protocol remains available for
direct callers; registration uses the capture-aware protocol.

`make_request` copies parameter/constant/option containers and validates their
immutable contents. `request_from_context` is the compatibility boundary: it copies
only signature metadata and literal constants from `SpecializeContext`, canonicalizes
shared dynamic dimension names, and keeps inexact extents dynamic. Declared dimensions
share by their DSL symbol literal, even through different Python aliases;
synthesized dimensions share only when they are the same instance,
so reusing a local name does not impose equality on independent extents. A slice whose
length is unproven receives an independent symbol even if metadata retains its parent's
dynamic dimension as a fallback; full slices preserve proven equality. Repeated aliases
of the same inexact tensor metadata share their axis symbols; independent views do not.
Shortened views retain separate runtime extent identities alongside their fallback
bounds. Full slices, whole-shape reads, and descriptor-based reshapes preserve these
identities, including when the derived view crosses an ordinary JIT dependency.
Omitted layouts normalize to ND, while unsupported constants fail explicitly.
Strides remain unconstrained and offsets
runtime-bound because the legacy context cannot prove those values.

Existing `__pypto_jit__()` hooks may return a stable `JITFunction`; discovery checks
the return type and leaves compilation to the ordinary JIT path. `legacy_builder(ref,
target, lower)` adapts a context-based builder callback to capture/prepare, then calls
an explicitly supplied `lower(PreparedKernel)` to produce the legacy builder result.
The provider never sees the live context, AST, globals or tensor addresses. This is
an adapter boundary for the builder path in PR #3231; it does not install that PR's
runtime wrappers or make arbitrary registered objects executable in JIT programs.
Automatic generic call lowering and real DSL migration remain subsequent tasks.

## Requests and prepared kernels

`KernelRequest` carries ordered `ParameterSpec` records, compile-time constants,
a `CompileTarget` and effective options. `TensorSpec` records dtype, shape,
layout, strides, offset and alignment. Dimensions may be nonnegative integers
or request-local symbol names. Strides and offsets are in elements; an empty
stride tuple is unconstrained, not contiguous. `offset=None` means runtime-bound.
A scalar has an explicit `DataType`. Task IDs are not device scalar arguments.

Records are frozen dataclasses containing immutable tuples/frozensets. The
validators reject mutable nested inputs, duplicate names and non-finite constants.
Tuple-declared fields require actual tuples, including ABI parameters, tensor
dimensions, result bindings and include paths. An immutable set cannot substitute
for an ordered tuple. Constant/option entries also require `(name, value)` tuples;
capability sets remain unordered frozensets.
Container members and scalar metadata must match their declared types too.
Bytes are accepted only in `SourceFile.content`; malformed metadata fails with
`ContractError` before semantic validation or digest serialization.
Do not pass `SpecializeContext`, AST nodes, Python source expressions or actual
data pointers to a provider as normalized signature data.

`ExternalKernelProvider.prepare(kernel, request)` accepts an opaque provider-owned
reference and returns `PreparedKernel`:

| Field | Contract |
| ----- | -------- |
| `request` | Exact normalized input, including target/toolchain/options |
| `contract` | Logical parameters, effects, public returns, writebacks, alias permissions |
| `artifact` | Owned source/header bytes, bundle-relative paths and SHA-256 digests, entry, symbol, ordered include paths |
| `native_abi` | Independently versioned physical argument representation |
| `execution` | Core members, lanes, launch requirements and completion guarantee |
| `identity` | Provider/adapter versions, source/dependency digest and completeness promise |

Each `SourceFile.content` owns immutable `bytes`, retained for the lifetime of the
bundle. Providers snapshot every source/header before returning, so
`RuntimeBridge.lower()` can consume a `PreparedKernel` after provider temporary
directories or cache leases are released. `ExternalTask.source` uses the same
self-contained representation for bridge-generated output.

`validate_prepared` checks each payload against its SHA-256 digest. The top-level
`artifact_manifest.json` name is reserved for the existing artifact store and
cannot be a payload or directory. Bridges materialize validated bytes into private
directories and enforce path/symlink containment on disk. Persistent publication,
restore and leases still use `ArtifactStore`; the bundle is not another cache.
File paths cannot be ancestors of other files: a bundle containing both `include`
and `include/header.h` is rejected regardless of enumeration order.
An include directory and all its ancestors must also be distinct from bundled
files; include directories may contain ordinary bundled headers.

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
For an accessed tensor using `DATA_POINTER`, runtime-bound or symbolic offsets
require an `OFFSET` field. Symbolic extents require `DIMENSION` fields, and
unconstrained or symbolic strides require `STRIDE` fields for the corresponding
axes. Fixed integer constraints are available during provider compilation.
A supported descriptor ABI carries these values in its descriptor; unused
(`Access.NONE`) pointers do not require view metadata slots.

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
Runtime-target fields follow the same exact schema checks as provider records.
Supported ABI entries must be `(str, int)` tuples in a frozenset, with nonempty
names and positive integer versions; booleans and floats cannot authorize an ABI
through Python's numeric equality. Runtime name and version must be nonempty.

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
and identity invalidation, source handoff after provider directory cleanup, payload
integrity and reserved manifest paths. `test_external_ir_semantics.py` checks normalized forms
through round-trip serialization, the default pass pipeline and orchestration
code generation: explicit public return plus hidden state writeback, inline,
zero/multi-trip loops, branches, AIC/AIV/MIX and a view-to-root dependency.

`test_external_registry.py` covers lazy loading, ambiguous selection, lifetimes,
immutable captures, concurrent loading and builder request isolation. The legacy
hook regression in `test_extern_kernel.py` runs a full JIT compile.

These are CPU compiler checks. They do not establish numerical device correctness,
automatic source-level writeback propagation, safe overlapping execution, or full
runtime completion behavior. Those require the subsequent integration tasks and
focused device tests. Existing extern APIs continue to use their compatibility path.
