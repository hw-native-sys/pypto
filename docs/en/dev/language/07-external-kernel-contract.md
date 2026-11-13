# External Kernel Contract (Internal v1)

This is the interface baseline for tasks 34.1–34.4 and [RFC #3293](https://github.com/hw-native-sys/pypto/issues/3293).
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
`compatibility.py`. Call normalization and signature variants are implemented in the JIT;
the TMR bridge and runtime binding checks are described below.

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
bounds. Whole-shape reads and descriptor-based reshapes keep those axes dynamic,
including across ordinary JIT dependencies. A descriptor read of a local view may
receive an independent symbol: the parser cannot always prove it equal to the
scalar that originally sized the view. Full subscript slices preserve proven equality.
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
Connecting arbitrary registered objects to executable runtime wrappers and real DSL
migration remain subsequent tasks.

## CANNBotDSL preparation (task 34.5, first stage)

`pypto.jit.external.providers.cannbotdsl.CANNBotDSLProvider` implements direct
`prepare(kernel, request)` using the optional package's public `inspect_kernel`,
`export_device` and `get_include` APIs. Importing the adapter does not import the
compiler. This stage supports A5/CCEC, explicit positional parameters, tensor and
runtime scalar descriptors, and named constexpr values. Tensor requests must use
ND layout with explicit positive strides and innermost stride 1; unconstrained
strides are rejected rather than assumed contiguous. Shared dimension symbols
remain shared. Extent-only symbols admit zero; symbols used in any stride have
minimum 1, including when also used as an extent. Compiler options are not yet accepted.

For example, a `(rows, 32)` tensor whose row stride is dynamic is described as
`TensorSpec(("rows", 32), DataType.FP32, strides=("pitch", 1))`. A runtime-bound
view offset remains in the request; the eventual bridge must apply it exactly
once when constructing the provider's first-valid-element pointer.

The result is a `PreparedKernel` with the named `cannbotdsl.device_callable` v1
descriptor ABI, preserving the export's physical argument order. Public returns
are empty. Every tensor write creates an internal writeback with unknown coverage,
so an access mask cannot discard dependencies on untouched data. AIC/AIV/MIX
participants come from the export, including two AIV lanes for MIX.
Scalar element tags in the logical runtime contract must match the request;
equal byte widths alone are insufficient. Malformed exports, including missing
`manifest` or `source` attributes, raise `ContractError` before handoff.

Preparation also validates the v1 descriptor layout: an 8-byte pointer at byte
offset 0, followed by packed int64 dynamic shape fields and then dynamic stride
fields, with 8-byte alignment. Static dimensions occupy no descriptor slots.
Field offsets, per-array slots, axes, total size and scalar kind/bit width must
agree with the request. Numeric metadata is type-sensitive; boolean and floating
point substitutes for integer offsets or sizes are rejected.

Logical contract schema 6 is checked separately because physical `-1` markers
cannot express dimension equality. Tensor element types, GM address space,
shape/stride expressions and dimension bounds must match the request exactly;
extra or stronger constraints are rejected. The supported view convention is
first-valid-element pointers with consumer-applied offsets and non-overlapping
positive strides. Cross-parameter write overlap is not supported. These checks
validate declarations; checking actual addresses, bounds and overlaps on each
invocation remains the runtime bridge's responsibility.

The owned artifact contains `device_callable.hpp`, the complete
`device_manifest.json`, and digest-checked copies of DSL extension headers. The
manifest retains descriptor layout, view constraints, state clobbers, exclusive
resources, cache-line write isolation, completion and binary-loading requirements.
CANN SDK headers remain a consumer toolchain dependency. Preparation runs afresh;
its identity is explicitly incomplete because no complete Python closure/input
snapshot exists. This provider is therefore not registered with the capture-only
registry and does not claim preparation-cache reuse.

**This stage does not enable calls from JIT programs or device execution.** The
existing `c_pointer` TMR bridge rejects this descriptor ABI. Subsequent 34.5 work
must implement a bridge that checks the complete manifest, lifetime-safe JIT
binding, and the relevant compatibility regressions. In particular, resource and
64-byte GM write-isolation requirements cannot be dropped when introducing the
runtime path. Device completion, numerical and performance acceptance remain 34.7.
The next runtime work must establish cache-line conflict serialization and
validate compiler stack requirements, alongside descriptor packing and resource
ownership. The existing text-only ELF checks alone do not establish these guarantees.
`test_cannbotdsl_provider.py` checks the handoff with a fake public exporter and,
when installed, the real public compiler export; it does not validate a binary.

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
`state` and `out`; only the latter is projected as `result`. The JIT now emits
those projections and writeback assignments before parsing and SSA conversion.
Existing inline and storage analysis propagate writes through aliases while keeping
helper-local name rebindings local. Void public calls still capture hidden updates.
Result projections become IR before passes run: slot numbers are not stable across
`NormalizeReturnOrder`.

The internal `SpecializeContext.external_contract` hand-off accepts a prepared
contract for an already bound extern declaration. Its public return mapping must
match the declaration: caller metadata is inferred before this hand-off, so a
different mapping is rejected rather than reusing stale result metadata.
Legacy externs derive the same
call plan from their validated direction annotations and public return aliases.
Partial/unknown writes become `InOut`; public return ordering, repeated aliases,
readonly aliases, and complete Python tuple annotations retain their meaning.
The native ABI's argument order is unchanged.
Writeback projections retain the actual argument's descriptor type, including
dynamic dimensions read through local aliases. Legacy InOut-only stubs with a
bare `-> pl.Tensor` annotation expose their first InOut buffer as the public result;
unannotated void stubs keep those writes hidden.

Calls must be standalone statements or the complete value of an assignment or
return. Nested external calls and calls inside conditions, loop iterators, or
context-manager headers fail with a diagnostic; bind their results explicitly.
View arguments are evaluated once. Calls in loop/branch bodies stay in those bodies,
so zero-trip loops and untaken branches do not issue a kernel.

## Call-site signature variants

Graphs containing extern kernels specialize each dependency, including ordinary
helpers, by constexpr values and tensor shape/dtype/layout plus scalar dtype.
Equivalent signatures reuse a generated function; incompatible signatures receive
separate names. Dynamic symbols are canonicalized by equality relationships rather
than spelling or sampled bounds. Independent shortened views stay independent.
Omitted layout stays distinct from explicit `ND`: omission can inherit a downstream
declaration, while explicit layout takes precedence. Local allocations retain their
known layout, including explicit or default `ND`.
Bare `pl.Scalar` helper/extern parameters inherit the call-site scalar type; explicit
scalar annotations keep their declared types.
Literal, explicitly typed `pl.const`, arithmetic, cast, dimension-read and local scalar arguments use the
parser's scalar promotion rules. Branches retain a scalar type only when both
paths agree.
Sample-argument binding carries those declared dtypes too; scalar values do not
split the cache. Missing or unsupported external parameter annotations name the
function and parameter in the diagnostic.

Function/call indices are request-local. One metadata walk per caller variant supplies
its call sites, including composed slice/reshape arguments, instead of replaying the
caller prefix at each call. The existing metadata walker handles branch joins.
Returned aliases of direct view arguments retain metadata for subsequent JIT calls.
An inline helper that directly returns an extern call preserves the same result
metadata as one that binds the result to a local before returning it.
Slices that read the same unchanged runtime count share an extent symbol; scalar
rebindings and uncertain control-flow joins invalidate that equality.

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
scheduling order. Known straight-line aliases and views passed to provider contracts
without the required permission fail during normalization unless constant one-dimensional
sibling slices prove disjoint intervals in the same parent coordinates. Unknown
coordinates, clamped slices, different parents, and reshaped coordinate systems remain conservative.
Name rebindings change
that provenance, including tuple/list unpacking, constant-index projections from
immutable tuple bindings, and keyword-based views. Uncertain
control-flow joins discard the proof. This static check
does not prove distinct input buffers or aliases across helper boundaries disjoint.
The runtime bridge must check those cases, including on cache hits. Legacy externs
retain their existing alias policy.

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

## TMR bridge (task 34.4)

`pypto.backend.tmr_bridge.TMRBridge` lowers `c_pointer` ABI version 1 for
`RuntimeTarget(runtime="tmr", version="1", ...)`, targeting A5/CCEC. This is an
internal runtime-interface version, not the runtime package's release number.
The provider supplies an includable device function returning `void`; a host
launch stub is not callable by this bridge. No optional DSL is imported.

The physical signature may reorder logical parameters. Tensor slots precede scalar
slots in TMR, independently of native argument order. Scalars preserve bits and
must have the declared native C++ type. Metadata uses unsigned 1/2/4/8-byte integers;
its declared units determine element-to-byte conversion. Data pointers name allocation
bases: the native function applies the separately passed or compile-time offset once.
Tensor alignment constrains that allocation-base pointer, independently of the offset;
it does not require the resulting view address to have the same alignment.
Compilation verifies native argument count, types, sizes and alignments.

`lower()` returns owned source bytes plus a self-contained host checker.
`with materialize(task) as native:` gives a private source lease;
`native.configure_context(context)` checks the prepared request and attaches sources,
includes, contract and checker to an existing extern specialization. Keep that lease
alive until the existing runtime has compiled/published its sources. AIC/AIV members,
mixed Groups and vector lane count come from the execution contract. Registered-object
discovery and CANNBotDSL migration remain task 34.5; persistent reuse remains task 34.6.

Orchestration invokes the checker after building actual task arguments and before
submission, including direct calls, explicit submits and reordered SPMD/Group wrappers.
It checks rank/dtype, fixed and shared extents, positive strides, offsets, alignment,
storage/address bounds and native metadata widths on each invocation. Unpermitted
write overlaps are rejected using conservative byte intervals; disjoint sibling views
and explicit alias permissions are supported. Checks remain in generated orchestration
on artifact hits. Before the native call, the wrapper disables atomics before saving
CTRL, so restoring control state cannot re-enable an inherited atomic mode. It clears
inherited HF32 mode on AIC cores; AIV preserves that control bit. It drains pipelines,
writes back caches and applies the DDR barrier before returning, then restores control
state with atomics disabled.

Unsupported cases fail explicitly: descriptor ABIs, non-ND layouts, ranks outside
1–5, FP16/BF16 scalars, other architectures/runtimes, logical launch context,
synchronized/full-resident launches, runtime-allocated output slots and recorded
Graph replay. Allocate output storage before submitting. CPU tests compile and execute
argument conversion and checks against pinned TMR headers; mocked completion intrinsics
verify sequencing, not device visibility. Device acceptance remains task 34.7.

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

`test_external_normalization.py` starts from JIT source and exercises hidden writes
through nested inline helpers, control flow, partial-write contracts, views, local
rebindings, signature variants, and unsupported nested-call/overlap diagnostics.

These are CPU compiler checks. They do not establish numerical device correctness,
safe overlapping execution, or full
runtime completion behavior. Those require the subsequent integration tasks and
focused device tests. Existing extern APIs continue to use their compatibility path.
