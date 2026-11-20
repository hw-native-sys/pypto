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
view offset remains in the request; the call adapter below applies it exactly
once when constructing the provider's first-valid-element pointer. Runtime
binding validation must prove that the resulting address is in bounds.

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
validate declarations; the binding checker below validates actual addresses,
bounds and overlaps. A runtime bridge must invoke it before every submission.

The owned artifact contains `device_callable.hpp`, the complete
`device_manifest.json`, and digest-checked copies of DSL extension headers. The
manifest retains descriptor layout, view constraints, state clobbers, exclusive
resources, cache-line write isolation, completion and binary-loading requirements.
CANN SDK headers remain a consumer toolchain dependency. Preparation runs afresh;
its identity is explicitly incomplete because no complete Python closure/input
snapshot exists. This provider is therefore not registered with the capture-only
registry and does not claim preparation-cache reuse.

The A5 TMR bridge supports direct execution of the single-block AIC/AIV/MIX
subset described below. Automatic discovery and lifetime-safe binding in JIT
programs remain follow-up work; device numerical and performance acceptance
remain tracked separately. Preparation and binary compilation do not use caches.

### A5 AIC/AIV/MIX runtime path

`TMRBridge.lower(prepared, target)` accepts `cannbotdsl.device_callable` v1 for
TMR v1 on `a5/ccec`. It supports one AIC, one AIV, or one hardware-paired
AIC + two AIV group (MIX). It requires no implicit workspace and the exporter's exact resource, state,
loading and completion declarations. Static, disjoint allocations in AIV UB and
AIC L1/L0A/L0B/L0C are supported. The bridge validates integer sizes and offsets,
space/core ownership, alignment (32 bytes for UB, 512 for Cube spaces), capacity,
and exact per-space high-water marks. Limits are 248 KiB UB, 512 KiB L1,
64 KiB each L0A/L0B, and 256 KiB L0C. Dynamic allocations, overlapping allocation
records, shared SSBUF and other memory spaces remain unsupported. Unknown or stronger
requirements fail before compilation. Each submission must contain exactly one
logical block without synchronized start.

TMR's persistent executor calls kernels synchronously and serially on each
physical core, including queued tasks. During the callback it does not use the
kernel's local memory or synchronization events. That execution interval gives
the kernel exclusive ownership; no global lock serializes independent cores.
The wrapper drains inherited pipelines and scalar GM writes before entry and
forces atomic-none. After the exported call it drains again, flushes scalar GM,
executes a DDR barrier, restores ASC/DMA/padding/quantization/mask defaults and
the saved CTRL, and returns with atomic-none. Only then may the executor publish
a participant FIN or execute the next callback. AIC entry also clears inherited
HF32 mode; both compiled variants restore the saved CTRL. TMR places a MIX task
on one physical cluster and completes the task only after all three participant
FINs. Its existing completion counter supplies the group join; a local pipe
barrier is not treated as a cross-core barrier. The kernel author must still
ensure intra-group synchronization and disjoint cache-line ownership for writes.

Use `pypto.backend.cannbotdsl_runtime.compile_task(task, compiler,
pto_isa_root=...)` to produce a tuple of text images in `task.prepared.execution.cores`
order: one image for AIC/AIV, or `(aic_image, aiv_image)` for MIX. Each image is
registered as a `CoreCallable`; both AIV lanes reuse the MIX AIV image. Every image
must pass admission before the tuple is returned. It takes C API
headers from the compiler's selected CANN SDK, retains owned source through
compilation, and requires text-only ELF admission and static zero-stack reports.
Text-only compilation omits optional frame-pointer bookkeeping so spill-free
SIMD VF calls can satisfy the same zero-stack check; actual stack use remains rejected.
The ordinary JIT `configure_context` route rejects this task's binary policy
rather than compiling it without admission checks.

The direct runtime system tests include FP16 Cube matmul with FP32 accumulation,
SIMD FP32 `(x + y) * x`, and a MIX task executing both pipelines. In the MIX
producer, each AIV lane owns one 128-element tile. The standalone SIMD task and
the dependent AIV each process two tiles (256 elements) on one AIV; the dependent
task consumes both MIX outputs after whole-group completion. Each tile uses two
64-element register operations because an A5 vector register holds 256 bytes.
Repeated runs change inputs and poison outputs to expose stale data.
These cases exercise allocated on-chip buffers and real compute instructions;
they do not yet test intra-group Cube-to-vector buffer handoffs. Compile-only
mode builds both device and orchestration binaries; numerical acceptance requires
running the same tests on an A5 device.

`build_submitter(tasks, function_ids)` returns an owned orchestration header.
Pass one function-ID tuple per task, in the same order as its compiled images;
for example `((10,), (20, 21), (30,))` registers an AIC → MIX → AIV batch.
IDs must be nonnegative and unique across distinct task/core images. Repeated
occurrences of an equal `ExternalTask` may reuse the same ID for the same core;
one ID mapped to different tasks or cores is rejected within the batch. Register
each compiled kernel at its corresponding ID. The generated function takes `std::array<CoreTaskArgs*, N>` and returns the
submitted task IDs. Before any submission it checks all launches and bindings;
it then uses the ordered batch helper to disable early dispatch and add
completion-based `DEP_WAIT` edges for every conflicting 64-byte cache line.
Actual submission uses `rt_submit_task(MixedKernels, args)`. Unused slots are -1;
MIX fills AIC with the first ID and both AIV slots with the second ID, yielding
one task ID whose completion covers the whole group.
Materialize all header files for the orchestration build and retain callables,
argument descriptors and tensor storage until execution completes. Calls outside
this batch need their own dependencies or GM isolation, including PyPTO producers
and consumers.
Include the returned `SourceArtifact.entry` rather than a fixed filename. Runtime
entry filenames include a digest of the batch and registration IDs, so multiple
submitters can share one materialization directory and translation unit.

Host tests exercise the wrapper's drain/restore order and actual TMR submission
API plus the pinned scheduler's whole-group completion counter; optional CCEC
tests compile static/dynamic AIC/AIV/MIX exports through admission.
The focused runtime test `tests/st/runtime/external_kernel/test_cannbotdsl_runtime.py`
builds dependent chains of each topology and an AIC → MIX → AIV chain, checking
repeated numerical results on A5. MIX test participants write separate cache lines.
`--codegen-only` compiles device and orchestration binaries and constructs
callables without launching a device. Both modes require the pinned Simpler
package and CANN toolchain: `KernelCompiler` uses `simpler_setup` even before
callable construction. The `tests/st/codegen` CI job does not install Simpler or
CANN and does not run this direct-runtime test. Compile-only results do not
establish device completion visibility or performance acceptance.

### Text-only binary admission

`KernelCompiler.compile_incore(..., text_only=True)` provides an opt-in A5
compilation path for device-callable sources with a `kernel_entry` wrapper.
It emits separate function/data sections and a compiler stack-usage report,
then links with entry ordering and garbage collection. This removes unreferenced
SDK data such as `g_tilingKey`.

Before returning the linked ELF, the compiler requires one nonempty executable
`.text` section, a global `kernel_entry` at its first byte, no remaining
relocations or unresolved symbols, and valid
section and symbol tables, including a null section at index 0. All section names
must be ASCII so the loader cannot select different bytes by discarding invalid
characters during name decoding. Additional allocated sections are not rejected
solely for being allocated and nonempty; this does not prove that discarded data
is unnecessary during execution. Conventional allocated data sections
(`.data`, `.rodata`, `.bss`, small-data and TLS variants) and NOBITS storage
are rejected for self-contained callees because text extraction discards them. Nonempty
`.ascend.meta*` sections are rejected because kernel launch metadata can require
additional SU/SIMT stack or UB provisioning.
Every retained function must have a compiler report
showing **static zero-stack usage**. Function aliases with identical nonempty code
ranges share a frame and need at least one report; every available alias report
must show static zero-stack usage. Zero-sized symbols are checked independently.
Missing reports, dynamic frames and positive
stack sizes fail: the persistent runtime currently exposes no per-task compiler
stack reservation contract. Reports for functions discarded by the linker do
not consume a budget. Compilation, report generation and checking happen in the
same private build directory, which is removed on success or failure.

This mode supports A5 AIC/AIV compilation and rejects simulation and separately
compiled VPTO helpers, whose stack evidence is not covered. Ordinary compilation
keeps its existing behavior. The check validates compiler-produced binary
structure; it does not prove arbitrary machine code position-independent.
It does not enable descriptor JIT calls, reserve local memory/synchronization
resources, enforce batch ordering or establish device completion visibility.
The bridge must still establish those guarantees before submission. A future
artifact-cache integration must include this build policy in its identity and
retain admission evidence; this direct compile path performs no cache lookup.

`test_kernel_compiler.py` covers invalid images and stack reports, failure cleanup,
and real CANNBotDSL static/dynamic exports compiled and linked with CCEC when
available. These are binary admission checks, not device execution acceptance.

### CANNBotDSL call adapter (compile-only)

`pypto.backend.cannbotdsl_abi.build_call_adapter(prepared)` produces an owned
`SourceArtifact` containing `call_adapter.hpp` and the native source, manifest
and headers. It reads no provider directory and imports no optional compiler.
The adapter revalidates the manifest against the prepared contract, native ABI,
execution metadata and source/header digests before generating code.

The generated device function consumes TMR argument storage (tensors first,
then scalar bit slots), preserves the export's physical argument order, applies
element offsets once, and populates dynamic shape/stride arrays. C++ assertions
check the actual function signature, POD layout, pointer and array types,
offsets, sizes and alignment against the manifest. Integer carriers preserve
bits even when their native signedness differs from the logical type. Tensor
element types must match the request; equal byte widths are insufficient. Floating
scalars preserve signed zero and NaN payloads. Bool slots normalize the low byte
to a valid C++ bool. TMR limits are 32 tensors, 16 scalars and tensor rank 1..5;
FP16/BF16 scalar slots are not supported.

This is a callable header, **not an `ExternalTask` or a loadable runtime entry**:
it deliberately defines no `kernel_entry`. The shared runtime bridge above supplies
that entry and the required execution policy. Its caller must run the binding checker on every invocation and meet
the retained resource, cache-line, stack, binary-loading and completion
requirements. Host C++ tests execute packing and offset handling; optional
CCEC tests compile real public exports. Neither establishes device execution
or runtime completion correctness.

### CANNBotDSL per-invocation binding checker

`pypto.backend.cannbotdsl_abi.build_binding_checker(prepared)` produces an owned
`SourceArtifact` containing `binding_check.hpp`. After revalidating the prepared
manifest and source/header digests, it generates a checker for actual TMR
`CoreTaskArgs`. Include it with the A5 TMR orchestration headers and invoke its
exported symbol before packing or submitting those same bindings. Generation
does not import the optional compiler or capture any live tensor addresses.
The checker must run again on every call, including artifact cache hits.

It reuses the TMR binding checks for argument counts, existing output storage,
rank/dtype, static and shared dimensions, positive strides, offset constraints,
storage bounds and arithmetic overflow. CANNBotDSL alignment applies to the
first-element address (`buffer.addr + start_offset * element_bytes`), with at
least natural element alignment. The `c_pointer` bridge retains its separate
allocation-base alignment rule.

Within each nonempty tensor, the checker sorts axes by stride, ignores singleton
axes, and requires each step to clear the span of previously visited axes. This
proves non-overlap for padded and permuted layouts; interleaved layouts that
cannot be proved by this conservative check are rejected. Across parameters,
bounding byte intervals may overlap only when neither parameter writes. Empty
views touch no bytes but still require valid strides, offsets and representable,
aligned pointers. These checks remain active in release builds.

TMR metadata limits are enforced before generating the checker. `INDEX` uses
the runtime `INT64` tensor tag; E4M3FN and E8M0 use their distinct FP8 tags.
E5M2 is rejected because TMR has no corresponding tag. FP8 runtime scalars and
FP16/BF16 runtime scalars are unsupported.

Passing this checker **does not authorize device execution**: byte-level
non-overlap does not establish 64-byte cache-line isolation between tasks.
Resource ownership, stack/loading validation and task completion require the
shared runtime bridge above. Automatic JIT binding remains follow-up work.

### CANNBotDSL cross-invocation cache-line conflicts

`pypto.backend.cannbotdsl_abi.build_conflict_checker(first, second)` produces an
owned `conflict_check.hpp` with both binding checkers. The two `PreparedKernel`
records may have different signatures, dtypes and access effects. Their manifests
must declare 64-byte `serialize_conflicting_cache_lines` GM isolation; missing or
unknown policies are rejected. The optional `within_group` field must equal
`kernel_author_must_ensure`: synchronization within a cooperative kernel group
remains the kernel author's obligation. The generated symbol accepts two `CoreTaskArgs`
objects and returns whether this pair requires serialization.

Both invocations are fully validated before the predicate returns, even when a
conflict is already apparent. It computes each accessed tensor's conservative
byte envelope from its actual allocation address, element offset, dimensions,
strides and element width, then compares inclusive cache-line indices. Using
inclusive indices avoids an overflowing round-up at the top of the address
space. Empty tensors and `Access.NONE` contribute no accesses. Read/read sharing
is allowed; a shared line with at least one write returns `true`. Holes in strided
views are included conservatively.

For example, two one-element FP32 output views at `0x2000` and `0x2004` do not
share bytes, but the predicate returns `true` because both touch line `0x2000`.
Moving the second view to `0x2040` returns `false` when their other accesses are
independent. A write sharing a line with the other invocation's input also
requires serialization. Pointer offsets are applied before computing lines.

This is a scheduling predicate, not a lock or submission API. A consumer must
compare every potentially concurrent pair and enforce completion-before-start
ordering whenever it returns `true`. Checking only neighboring calls is
insufficient. Bindings must stay unchanged and their storage must stay alive
until execution completes; changes require another check. A `false` result
proves only this pair's GM non-conflict and says nothing about local resources,
stack/loading requirements or completion visibility. Calls outside this protocol
also require a dependency or isolation policy. The batch helper below converts
these predicates into TMR dependencies; JIT integration remains follow-up work.

### CANNBotDSL ordered batch submission

`pypto.backend.cannbotdsl_submit.build_ordered_submitter(prepared)` accepts a
nonempty tuple of prepared kernels and produces an owned header at
`SourceArtifact.entry`. Its filename includes the batch identity so different
batches do not overwrite one another; identical support headers can be shared.
Its generated function takes `std::array<CoreTaskArgs*, N>` and a synchronous
callback `(size_t index, CoreTaskArgs& args) -> TaskId`, returning the submitted
IDs in batch order. The callback must submit exactly the indexed prepared
kernel, synchronously consume the provided arguments and dependencies, and
return its fresh task ID. This helper does not load kernels or authorize their
execution; the callback's runtime integration must establish those guarantees.

Before the first callback, the helper validates every argument and existing
dependency, checks every pair, and allocates dependency storage. Every
conflicting earlier call contributes `DEP_WAIT`, including nonadjacent calls.
Disjoint and read/read pairs add no edge. Existing dependencies keep their
flags, including `DEP_RETAIN`; a matching edge gains `DEP_WAIT` without losing
retention. A batch with outputs on lines A, B, A therefore adds first-to-third
ordering while allowing the middle task to remain independent.

The helper temporarily binds its dependency arrays to each `CoreTaskArgs` and
disables early dispatch. Original dependency pointers, flags and the early
setting are restored after each callback, including during exception unwinding.
The callback must not retain these temporary pointers or mutate batch bindings.
A failed/invalid or reused task ID stops subsequent submission. Already submitted
tasks cannot be rolled back; a callback failure is not an atomic batch failure.

All producers must remain referenceable in the same active TMR scope until
consumers are submitted. Bindings must stay fixed and backing storage must
survive execution. Every invocation recomputes conflicts; no address-dependent
result is cached. The helper orders only this batch: other potentially concurrent
calls need separate ordering or isolation. It does not prove intra-group safety,
resource ownership, stack limits, binary loading or completion visibility by
itself. The direct runtime path described above composes this helper with
`TMRBridge.lower`, checked compilation and actual TMR submission for the supported
subset. Automatic JIT discovery and binding remain follow-up work.

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

## TMR c_pointer bridge (task 34.4)

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
discovery remains task 34.5; persistent reuse remains task 34.6. CANNBotDSL uses
the dedicated direct runtime path documented above.

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

This c_pointer path rejects descriptor ABIs; the dedicated runtime path handles the
supported CANNBotDSL subset. Other unsupported cases fail explicitly: non-ND layouts, ranks outside
1–5, FP16/BF16 scalars, other architectures/runtimes, logical launch context,
synchronized/full-resident launches, runtime-allocated output slots and recorded
Graph replay. Allocate output storage before submitting. CPU tests compile and execute
argument conversion and checks against pinned TMR headers; mocked completion intrinsics
verify sequencing, not device visibility. Device acceptance remains task 34.7.

## Raw device-image entry

Hardware kernels (A2/A3 and A5, AIC and AIV) are loaded as raw `.text` bytes.
`kernel_entry` must be at offset zero of a nonempty executable `.text` section.
The linker option `-e kernel_entry` selects ELF entry metadata; it does not move
the function. A source wrapper that retains an earlier out-of-line function can
therefore link successfully but violate the loader ABI.

PyPTO validates the linked AI Core ELF before returning it from `compile_incore`
and validates source-adjacent cached ELFs again before extraction or cache
publication. Nonzero entry offsets fail with the entry address, `.text` address
and offset. Link the entry first or remove preceding out-of-line functions;
verify the resulting ELF rather than relying on source order. CPU simulator
shared libraries resolve `kernel_entry` by symbol and do not use this check.

Ordinary extern validation does not impose the VPTO helper prohibition on other
allocated sections, such as AscendC block-local storage. VPTO's additional
self-contained-image checks remain enabled for helper builds.

Mutable binary-cache contexts use schema 3: older stamps invalidate both raw
payloads and cached ELF sidecars, even when their hashes match. Published
`BINARY_READY` manifests use schema 2, including distributed parent and child
manifests. Older prebuilt manifests are rejected before callable construction;
rebuild those artifacts from source. Raw bytes alone cannot prove where their
original ELF entry was located. New, validated cache entries remain reusable.

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

Self-contained callees still reject conventional allocated data sections (`.data`, `.rodata`, `.bss` and their variants) and NOBITS storage, which text extraction cannot preserve. Compiler metadata such as `__CCE_VF_STACK_SIZE` is not rejected merely for having SHF_ALLOC.
